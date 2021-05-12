using Random, DataFrames, CSV, Chain, StatsBase, BenchmarkTools

# 4 CSVs

const n = 100

Random.seed!(123)
df1 = DataFrame(
    x=sample(["A", "B", "C", "D"], n, replace=true),
    y=rand(n),
    z=randn(n),
)

Random.seed!(124)
df2 = DataFrame(
    x=sample(["A", "B", "C", "D"], n, replace=true),
    y=rand(n),
    z=randn(n),
)

Random.seed!(125)
df3 = DataFrame(
    x=sample(["A", "B", "C", "D"], n, replace=true),
    y=rand(n),
    z=randn(n),
)

Random.seed!(126)
df4 = DataFrame(
    x=sample(["A", "B", "C", "D"], n, replace=true),
    y=rand(n),
    z=randn(n),
)

dir = tempdir()

for (idx, df) in enumerate([df1, df2, df3, df4])
    df |> CSV.write("$(dir)/df$(idx).csv")
    # println("$(dir)/df$(idx).csv")
end

files = filter!(endswith(".csv"), readdir(dir, join=true))

@btime reduce(vcat, CSV.read(file, DataFrame) for file in $files)

@btime mapreduce(DataFrame ∘ CSV.File, vcat, $files)
