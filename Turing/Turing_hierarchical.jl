using Turing, RDatasets
using Random:seed!
using Statistics: mean, std
using LinearAlgebra: cholesky, kron
using MLDataUtils:rescale!

seed!(1)

mtcars = RDatasets.dataset("datasets", "mtcars")

#### Varying Intercept Model ####

# Data prep
y = mtcars[:, :MPG]
idx = mtcars[:, :Cyl] # vector of group indeces
idx = map(idx) do i
    i == 4 ? 1 :
    i == 6 ? 2 :
    i == 8 ? 3 : missing
end
X = Matrix(select(mtcars, [:HP, :WT])) # the model matrix
μ, σ = rescale!(X; obsdim=1)

# Model
@model varying_intercept(X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2)) = begin
    # priors
    μ ~ Normal(mean(y), 2.5 * std(y))       # population-level intercept
    σ ~ Exponential(1 / std(y))             # residual SD
    # Coefficients Student-t(ν = 3)
    β ~ filldist(TDist(3), predictors)
    # Prior for variance of random intercepts. Usually requires thoughtful specification.
    σᵢ ~ Truncated(Cauchy(0, 2), 0, Inf)
    μᵢ ~ filldist(Normal(0, σᵢ), n_gr)      # group-level intercepts

    # likelihood
    ŷ = μ .+ X * β .+ μᵢ[idx]
    y ~ MvNormal(ŷ, σ)
end

model = varying_intercept(X, idx, y)

@time chn = sample(model, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)

#### Varying Slope Model ####

# Insert Column of 1's as intercept
X = [fill(1, size(X, 1)) X]

# Model
@model varying_slope(X, idx, y) = begin
    n_gr = length(unique(idx))                          # number of groups
    K = size(X, 2)                                      # number of columns in X

    # priors
    # μ ~ Normal(mean(y), 2.5 * std(y))                   # population-level intercept
    β ~ filldist(LocationScale(0, 10, TDist(3)), K)     # group-level slopes
    σ ~ Exponential(1 / std(y))                         # residual SD

    # likelihood
    ŷ = X .* β
    y ~ MvNormal(ŷ, σ)
    return β
end

model = varying_slope(X, idx, y)

chn = sample(model, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)


#### Varying Intercept-Slope Correlated Model ####

# Insert Column of 1's as intercept
X = [fill(1, size(X, 1)) X]

# Model
@model varying_slope(X, idx, y) = begin
    n_gr = length(unique(idx))
    m_X = size(X, 2)

    # priors
    β ~ filldist(TDist(3), m_X)                               # hyperprior for intercept and slope
    σ ~ Exponential(1 / std(y))                               # residual SD
    Ω ~ LKJ(m_X, 2.0)                                         # Correlation Matrix
    τ ~ filldist(Truncated(Cauchy(0, 2), 0, Inf), size(X, 2)) # SD for intercept and slope
    z_ρ ~ filldist(Normal(), m_X, n_gr)                       # matrix of intercepts and slope

    # build Σ and Random Effets
    L = LinearAlgebra.cholesky(Ω).L
    D = I(m_X) .* LinearAlgebra.kron(ones(m_X)', τ)           # Diagonal(τ)
    z = D * L * z_ρ                                           # non-centered version of β

    # likelihood
    ŷ = β[1] .+ z[1, idx] .+ (β[2] .+ z[2, idx]) .* X[:, 2]
    y ~ MvNormal(ŷ, σ)

    # generated quantities
    return z
end

model = varying_slope(X, idx, y)

chn = sample(model, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)

qtd = generated_quantities(model, chn);
mean(qtd)
