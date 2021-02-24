using Turing, DynamicPPL, DynamicHMC, StatsPlots

N = 10_000  # 10,000 obs
K = 7      # 7 predictors
X = Matrix{Float64}(undef, N, K);
for i = 1:K
    X[:, i] = randn(N) + rand(0:100, N)
end
y = rand(-100:100, N) + (X * rand(7));

@model big_data(N, y, X) = begin
    # Priors
    β ~ MvNormal(zeros(size(X, 2)), sqrt(10))
    σ ~ Exponential(1)

    # Likelihood
    y ~ MvNormal(X * β, σ)
end

model = big_data(N, y, X)

# 56s
@time chn = sample(model, NUTS(), MCMCThreads(), 2_000, 4)
