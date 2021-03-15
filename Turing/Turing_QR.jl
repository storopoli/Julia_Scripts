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

#### Varying Intercept Model ####
# Model
@model varying_intercept(X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2)) = begin
    # priors
    μ ~ Normal(mean(y), 2.5 * std(y))       # population-level intercept
    σ ~ Exponential(1 / std(y))             # residual SD
    # Coefficients Student-t(ν = 3)
    β ~ filldist(TDist(3), predictors)
    # Prior for variance of random intercepts. Usually requires thoughtful specification.
    τ ~ truncated(Cauchy(0, 2), 0, Inf)
    αⱼ ~ filldist(Normal(0, τ), n_gr)      # group-level intercepts

    # likelihood
    ŷ = μ .+ X * β .+ αⱼ[idx]
    y ~ MvNormal(ŷ, σ)
end

model = varying_intercept(X, idx, float(y))
model_qr = varying_intercept(Matrix(Q), idx, float(y))

# 135.6s
@time chn = sample(model, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)

# 19.6s
@time chn_qr = sample(model_qr, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)

#### get β from Q^T * y by R^-1 * β ####
beta = mapslices(x -> R^-1 * x, chn_qr[:,namesingroup(chn_qr, :β),:].value.data, dims=[2])
chn_qr_reconstructed = hcat(Chains(beta, ["displ", "year"]), chn_qr)

#### ESS Comparison ####
# 4238 ESS
combine(DataFrame(ess(group(chn, :β))),  [:ess, :rhat] .=> mean)
# 1614 ESS
combine(DataFrame(ess(group(chn, :αⱼ))),  [:ess, :rhat] .=> mean)

# 6004 ESS
combine(DataFrame(ess(group(chn_qr, :β))),  [:ess, :rhat] .=> mean)
# 1688 ESS
combine(DataFrame(ess(group(chn_qr, :αⱼ))),  [:ess, :rhat] .=> mean)

#### NCP Varying Intercept Model ####
@model varying_intercept_ncp(X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2)) = begin
    # priors
    μ ~ Normal(mean(y), 2.5 * std(y))       # population-level intercept
    σ ~ Exponential(1 / std(y))             # residual SD
    # Coefficients Student-t(ν = 3)
    β ~ filldist(TDist(3), predictors)
    # Prior for variance of random intercepts. Usually requires thoughtful specification.
    τ ~ truncated(Cauchy(0, 2), 0, Inf)
    zⱼ ~ filldist(Normal(0, 1), n_gr)      # NCP group-level intercepts

    # likelihood
    ŷ = μ .+ X * β .+ zⱼ[idx] .* τ
    y ~ MvNormal(ŷ, σ)
end

model_ncp = varying_intercept(X, idx, float(y))
model_qr_ncp = varying_intercept(Matrix(Q), idx, float(y))

# 138.6s
@time chn = sample(model_ncp, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)

# 14.9s
@time chn_qr = sample(model_qr_ncp, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)

#### get β from Q^T * y by R^-1 * β ####
quantiles_beta = select(DataFrame(quantile(group(chn_qr, :β))), r"%")
mapcols(x -> R^-1 * x, quantiles_beta)

#### ESS Comparison ####
# 4398 ESS
combine(DataFrame(ess(group(chn, :β))),  [:ess, :rhat] .=> mean)
# 2198 ESS
combine(DataFrame(ess(group(chn, :αⱼ))),  [:ess, :rhat] .=> mean)

# 6149 ESS
combine(DataFrame(ess(group(chn_qr, :β))),  [:ess, :rhat] .=> mean)
# 1523 ESS
combine(DataFrame(ess(group(chn_qr, :αⱼ))),  [:ess, :rhat] .=> mean)
