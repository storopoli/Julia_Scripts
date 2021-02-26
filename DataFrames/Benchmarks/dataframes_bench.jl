using BenchmarkTools, DataFrames, Random
Random.seed!(1)

n = 10^8

df = DataFrame(
  group=rand(["A", "B", "C", "D", "E"], n),
  var_1=randn(n),
  var_2=rand(n) .* 1.5,
  var_3=rand(n)
)

Base.summarysize(df) / 1e9
# 3.2GB

@benchmark filter(:var_1 => <(0), df) # 868 ms ()
@benchmark transform(df, :var_1 => x -> x .* 2, :var_2 => x -> x.^2, :var_3 => x -> exp.(x)) # 3.593 s
@benchmark combine(groupby(df, :group), :var_1 => mean, :var_2 => mean, :var_3 => mean) # 2.753 s
