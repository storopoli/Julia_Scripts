# http://sherrytowers.com/2018/03/07/logistic-binomial-regression/
using Turing, CSV, DataFrames, LazyArrays, Plots, StatsPlots

df = CSV.read("Turing/StatisticalRethinking/oring.csv", DataFrame; header=2)

# there are only 6 O-rings in total
df.num_orings = 6 .- df.num_failure;

@model binomial_model(num_orings, temp, num_failure) = begin
    α ~ TDist(3)
    β ~ TDist(3)
    num_failure ~ arraydist(LazyArray(@~ BinomialLogit.(num_orings, α .+ temp .* β)))
end

model = binomial_model(df.num_orings, df.temp, df.num_failure)

chn = sample(model, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4, progress=true)

# function to convert logodds
function logodds2prob(logodds::Float64)
    return exp(logodds) / (1 + exp(logodds))
end

temps = 20:0.01:80
alpha, beta = quantile(chn)[:, :var"50.0%"]
y = logodds2prob.(alpha .+ beta .* temps) .* 6
alpha_l, beta_l = quantile(chn)[:, :var"25.0%"]
alpha_h, beta_h = quantile(chn)[:, :var"75.0%"]
lower = logodds2prob.(alpha_l .+ beta_l .* temps) .* 6
upper = logodds2prob.(alpha_h .+ beta_h .* temps) .* 6

plot(
    temps, y,
    linewidth=3,
    linecolor=:red,
    label="Predicted",
    xlabel="Temperature Fᵒ",
    ylabel="O-Rings Failures",
    ylims=(0, 6))
plot!(
    temps, lower,
    fillrange=upper,
    fillalpha=0.2,
    c=1, label="Credible 25% - 75% Band")
scatter!(
    df.temp, df.num_failure,
    markercolor=:blue,
    label="Actual")
vline!([28], c=:green, label="Temperature at Launch", linewidth=3)

