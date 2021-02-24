# http://sherrytowers.com/2018/03/07/logistic-binomial-regression/
using Turing, CSV, DataFrames, LazyArrays, Plots

df = CSV.read("Turing/StatisticalRethinking/oring.csv", DataFrame; header=2)

# there are only 6 O-rings in total
df.num_orings = 6 .- df.num_failure

@model binomial_model(num_orings, temp, num_failure) = begin
    α ~ TDist(3)
    β ~ TDist(3)
    num_failure ~ arraydist(LazyArray(@~ BinomialLogit.(num_orings, α .+ temp .* β)))
end

model = binomial_model(df.num_orings, df.temp, df.num_failure)

chn = sample(model, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)

temps = 10:80
avg, beta = median(chn)[:, :mean]
y = avg .+ exp.(beta .* temps)


loess_ = loess(temps, y);

scatter(df.temp, (df.num_failure ./ 6), markercolor=:blue, label="Actual")
plot!(temps, y, linecolor=:red, label="Predicted")
