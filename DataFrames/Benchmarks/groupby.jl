using Random, StatsBase, DataFrames, BenchmarkTools, Chain, Pipe
Random.seed!(123)

n = 10_000

df = DataFrame(
    x=sample(["A", "B", "C", "D"], n, replace=true),
    y=rand(n),
    z=randn(n),
)

# 420µs (0.42ms) (292µs M1 1.7.0, 534µs M1 Intel)
@benchmark @pipe $df |> groupby(_, :x) |> combine(_, :y => median, :z => mean)

# 409µs (0.4ms) (292µs M1 1.7.0, 534µs M1 Intel) (410µs Dell G5, 378µs using MKL and BLAS.set_num_threads)
@benchmark @chain $df begin
    groupby(:x)
    combine(:y => median, :z => mean)
end

# 506µs (0.5ms)
@benchmark combine(groupby($df, :x), :y => median, :z => mean)
