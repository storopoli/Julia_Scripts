@everywhere using Turing, DataFrames, Chain, CSV, HTTP, Distributed
@everywhere using Random:seed!
@everywhere using Statistics: mean, std

setprogress!(true)

addprocs()
@everywhere using Turing, DataFrames

@everywhere df = @chain HTTP.get("https://github.com/selva86/datasets/blob/master/mpg_ggplot2.csv?raw=TRUE") begin
    _.body
    CSV.read(DataFrame)
end

#### Data Prep ####
@everywhere idx_map = Dict(key => idx for (idx, key) in enumerate(unique(df.class)))
@everywhere y = df[:, :hwy]
@everywhere idx = getindex.(Ref(idx_map), df.class)
@everywhere X = Matrix(select(df, [:displ, :year]))

#### NCP Varying Intercept Model ####
@everywhere @model varying_intercept_ncp(X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2)) = begin
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

@everywhere model_ncp = varying_intercept_ncp(X, idx, float(y))

# 123s
@time chn2 = sample(model_ncp, NUTS(1_000, 0.65), MCMCDistributed(), 2_000, 4)
