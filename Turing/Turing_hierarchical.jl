using Turing, DataFrames, Pipe, CSV, HTTP, Zygote, ReverseDiff
using Random:seed!
using Statistics: mean, std

seed!(1)
setprogress!(true)

df = @pipe HTTP.get("https://github.com/selva86/datasets/blob/master/mpg_ggplot2.csv?raw=TRUE").body |>
    CSV.read(_, DataFrame)

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
    σⱼ ~ truncated(Cauchy(0, 2), 0, Inf)
    μⱼ ~ filldist(Normal(0, σⱼ), n_gr)      # group-level intercepts

    # likelihood
    ŷ = μ .+ X * β .+ μⱼ[idx]
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
    σⱼ ~ truncated(Cauchy(0, 2), 0, Inf)
    zⱼ ~ filldist(Normal(0, 1), n_gr)      # NCP group-level intercepts

    # likelihood
    ŷ = μ .+ X * β .+ zⱼ[idx] .* σⱼ
    y ~ MvNormal(ŷ, σ)
end

model_ncp = varying_intercept_ncp(X, idx, float(y))
prior_ncp = sample(model_ncp, Prior(), MCMCThreads(), 2_000, 4)

# 164s
@time chn2 = sample(model_ncp, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)

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
