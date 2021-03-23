using Turing, DataFrames, Chain, CSV, HTTP
using Random:seed!
using Statistics: mean, std
using LinearAlgebra:qr

seed!(1)
setprogress!(true)

df = @chain HTTP.get("https://github.com/selva86/datasets/blob/master/mpg_ggplot2.csv?raw=TRUE") begin
    _.body
    CSV.read(DataFrame)
end

#### Data Prep ####
idx_map = Dict(key => idx for (idx, key) in enumerate(unique(df.class)))
y = df[:, :hwy]
idx = getindex.(Ref(idx_map), df.class)
X = Matrix(select(df, [:displ, :year])) # the model matrix

#### QR Decomposition ####
Q, R = qr(X)
Q_ast = Matrix(Q) * sqrt(size(X, 1) - 1)
R_ast = R / sqrt(size(X, 1) - 1)

#### Varying Intercept Model ####
# Model
@model varying_intercept(X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2)) = begin
    # priors
    μ ~ Normal(mean(y), 2.5 * std(y))       # population-level intercept
    σ ~ Exponential(1 / std(y))             # residual SD
    # Coefficients Student-t(ν = 3)
    θ ~ filldist(TDist(3), predictors)      # coefficients on Q_ast
    # Prior for variance of random intercepts. Usually requires thoughtful specification.
    τ ~ truncated(Cauchy(0, 2), 0, Inf)
    αⱼ ~ filldist(Normal(0, τ), n_gr)      # group-level intercepts

    # likelihood
    ŷ = μ .+ X * θ .+ αⱼ[idx]
    y ~ MvNormal(ŷ, σ)
end

model = varying_intercept(X, idx, float(y))

# 90s Dell G5
@time chn = sample(model, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)

#### NCP Varying Intercept Model ####
@model varying_intercept_ncp(X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2)) = begin
    # priors
    μ ~ Normal(mean(y), 2.5 * std(y))       # population-level intercept
    σ ~ Exponential(1 / std(y))             # residual SD
    # Coefficients Student-t(ν = 3)
    θ ~ filldist(TDist(3), predictors)
    # Prior for variance of random intercepts. Usually requires thoughtful specification.
    τ ~ truncated(Cauchy(0, 2), 0, Inf)
    zⱼ ~ filldist(Normal(0, 1), n_gr)      # NCP group-level intercepts

    # likelihood
    ŷ = μ .+ X * θ .+ zⱼ[idx] .* τ
    y ~ MvNormal(ŷ, σ)
end

model_ncp = varying_intercept_ncp(X, idx, float(y))
model_qr_ncp = varying_intercept_ncp(Q_ast, idx, float(y))

# 70s Dell G5
@time chn_ncp = sample(model_ncp, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)

# 9s Dell G4
@time chn_qr_ncp = sample(model_qr_ncp, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)

#### Divergences ####
sum(chn[:numerical_error])
sum(chn_ncp[:numerical_error])
sum(chn_qr_ncp[:numerical_error])

#### Aceptance Rate ####
mean(chn[:acceptance_rate])
mean(chn_ncp[:acceptance_rate])
mean(chn_qr_ncp[:acceptance_rate])
