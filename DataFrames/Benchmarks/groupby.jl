using Random, StatsBase, DataFrames
Random.seed!(123)

n = 10_000

df = DataFrame(
    x = sample(["A", "B", "C", "D"], n, replace = true),
    y = rand(n),
    z = randn(n),
)

# 547µs
@benchmark @pipe df |> groupby(_, :x) |> combine(_, :y => median, :z => mean)
