using Distributions, StatsPlots


dists = Iterators.product(-1:1, 0:2) |> collect

p = plot()
for i in dists
    plot!(Normal(i[1], i[2]), label="μ = $(i[1]) and σ = $(i[2])")
end
xlabel!("x"); ylabel!("Frequency")
current()
