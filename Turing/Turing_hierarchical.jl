using Turing, DataFrames, Chain, CSV, HTTP, Zygote, ReverseDiff
using Random:seed!
using Statistics: mean, std

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

prior = sample(model, Prior(), MCMCThreads(), 2_000, 4)

# 155.6s
@time chn = sample(model, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)


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

model_ncp = varying_intercept_ncp(X, idx, float(y))
prior_ncp = sample(model_ncp, Prior(), MCMCThreads(), 2_000, 4)

# 199s
@time chn2 = sample(model_ncp, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)

#### get αⱼ from zⱼ by zⱼ * τ ####
τ = summarystats(chn2)[:τ, :mean]
αⱼ = mapslices(x -> x * τ, chn2[:,namesingroup(chn2, :zⱼ),:].value.data, dims=[2])
chn_reconstructed = hcat(Chains(αⱼ, ["αⱼ[$i]" for i in 1:length(unique(idx))]), chn2)

#### Different Autodiffs ####

# Zygote - Too Long
Turing.setadbackend(:zygote)
@time chn_zygote = sample(model_ncp, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)

# Tracker - 471s
Turing.setadbackend(:tracker)
@time chn_tracker = sample(model_ncp, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)

# ReverseDiff - Too Long
Turing.setadbackend(:reversediff)
@time chn_tracker = sample(model_ncp, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)
