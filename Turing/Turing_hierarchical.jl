using Turing, RDatasets
using Random:seed!
using Statistics: mean, std
using LinearAlgebra: cholesky, kron

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

# Model
@model varying_intercept(X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2)) = begin
    # priors
    μ ~ Normal(mean(y), 2.5 * std(y))       # population-level intercept
    σ ~ Exponential(1 / std(y))             # residual SD
    # Coefficients Student-t(ν = 3)
    β ~ filldist(TDist(3), predictors)
    # Prior for variance of random intercepts. Usually requires thoughtful specification.
    σⱼ ~ Truncated(Cauchy(0, 2), 0, Inf)
    μⱼ ~ filldist(Normal(0, σⱼ), n_gr)      # group-level intercepts

    # likelihood
    ŷ = μ .+ X * β .+ μⱼ[idx]
    y ~ MvNormal(ŷ, σ)
end

model = varying_intercept(X, idx, y)

# 22.6s
@time chn = sample(model, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)


#### NCP Varying Intercept Model ####
@model varying_intercept_ncp(X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2)) = begin
    # priors
    μ ~ Normal(mean(y), 2.5 * std(y))       # population-level intercept
    σ ~ Exponential(1 / std(y))             # residual SD
    # Coefficients Student-t(ν = 3)
    β ~ filldist(TDist(3), predictors)
    # Prior for variance of random intercepts. Usually requires thoughtful specification.
    σⱼ ~ Truncated(Cauchy(0, 2), 0, Inf)
    zⱼ ~ filldist(Normal(0, 1), n_gr)      # NCP group-level intercepts

    # likelihood
    ŷ = μ .+ X * β .+ zⱼ[idx] .* σⱼ
    y ~ MvNormal(ŷ, σ)
end

model = varying_intercept_ncp(X, idx, y)

# 22.3s
@time chn2 = sample(model, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)

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
# WIP -- https://statisticalrethinkingjulia.github.io/TuringModels.jl/models/varying-slopes-cafe/
@model varying_intercept_slope(X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2)) = begin

    # priors
    ρ ~ LKJ(predictors + 1, 1.0)                                      # Correlation Matrix
    σ ~ truncated(Cauchy(0, 2), 0, Inf)                               # residual SD
    σⱼ ~ filldist(Truncated(Cauchy(0, 2), 0, Inf), (predictors + 1))  # Hyperprior SD for intercepts and slopes
    β ~ filldist(Normal(0, 10), (predictors + 1))                     # Hyperprior intercept and slopes
    # α ~ Normal(0, 10)                                                # Hyperprior intercept
    # β ~ Normal(0, 10)                                                # Hyperprior for slopes

    dist_Σ = σⱼ .* ρ .* σⱼ'
    dist_Σ = (dist_Σ' + dist_Σ) / 2
    α_βⱼ ~ filldist(MvNormal(β, dist_Σ), 20)

    αⱼ = α_βⱼ[1, :]
    βⱼ = α_βⱼ[2:end, :]
    @show size(βⱼ)
    # build ρ and Random Effects
    # Ω = (Ω' + Ω) / 2
    # L = LinearAlgebra.cholesky(Ω).L
    # D = I(predictors) .* LinearAlgebra.kron(ones(predictors)', τ)   # Diagonal(τ)
    # z = D * L * z_ρ                                                 # non-centered version of β

    μ = αⱼ[idx] .+ X * βⱼ[idx]

    # likelihood
    # ŷ = μ .+ z[1, idx] .+ (β[2] .+ ) .* X[:, 2]
    y ~ MvNormal(μ, σ)
end

model = varying_intercept_slope(X, idx, y)

chn = sample(model, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)

qtd = generated_quantities(model, chn);
mean(qtd)
