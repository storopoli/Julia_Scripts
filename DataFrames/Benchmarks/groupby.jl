using Random, StatsBase, DataFrames, BenchmarkTools, Chain, Pipe
Random.seed!(123)

n = 10_000

df = DataFrame(
    x=sample(["A", "B", "C", "D"], n, replace=true),
    y=rand(n),
    z=randn(n),
)

# 520µs (0.52ms) (271µs M1 1.7.0)
@benchmark @pipe $df |> groupby(_, :x) |> combine(_, :y => median, :z => mean)

# 506µs (0.5ms) (268µs M1 1.7.0)
@benchmark @chain $df begin
    groupby(:x)
    combine(:y => median, :z => mean)
end

# 506µs (0.5ms)
@benchmark combine(groupby($df, :x), :y => median, :z => mean)
