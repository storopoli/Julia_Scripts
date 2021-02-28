using Turing, DynamicPPL, DynamicHMC, StatsPlots


J = 8
y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
σ = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
schools = [
    "Choate",
    "Deerfield",
    "Phillips Andover",
    "Phillips Exeter",
    "Hotchkiss",
    "Lawrenceville",
    "St. Paul's",
    "Mt. Hermon",
];

nwarmup, nsamples, nchains = 1000, 1000, 4;

@model school8(J, y, σ) = begin
    μ ~ Normal(0, 5)
    τ ~ truncated(Cauchy(0, 5), 0, Inf)
    θ ~ Normal(μ, τ)
    for j = 1:J
    y[j] ~ Normal(θ, σ[j])
end
end

model = school8(J, y, σ);

chn = sample(model, NUTS(), 2_000)

histogram(chn)
