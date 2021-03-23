using Turing, RDatasets, LazyArrays
using Random:seed!
using Statistics: mean, std
using MLDataUtils:rescale!
# We need a logistic function, which is provided by StatsFuns.
using StatsFuns:logistic

seed!(1)

default = RDatasets.dataset("ISLR", "Default")

# Convert "Default" and "Student" to numeric values.
default[!,:DefaultNum] = [r.Default == "Yes" ? 1.0 : 0.0 for r in eachrow(default)]
default[!,:StudentNum] = [r.Student == "Yes" ? 1.0 : 0.0 for r in eachrow(default)]

# Delete the old columns which say "Yes" and "No".
select!(default, Not([:Default, :Student]))

features = [:StudentNum, :Balance, :Income]
numerics = [:Balance, :Income]

# Rescale data
μ, σ = rescale!(default[!, numerics])

X = Matrix(default[:, features])
y = default[:, :DefaultNum]

# original formula: default ~ student + balance + income

lazyarray(f, x) = LazyArray(Base.broadcasted(f, x))
@model logreg(X, y; d=size(X, 2)) = begin
	μ ~ Normal(0, 2.5 * std(y))
	β ~ filldist(TDist(3), d)
	v = logistic.(μ .+ X * β)
	y ~ arraydist(lazyarray(Bernoulli, v))
end

model = logreg(X, y)

@time chn = sample(model, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)
