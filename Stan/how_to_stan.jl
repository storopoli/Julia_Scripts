using Stan, StanSample, StanDiagnose, Plots

model = read("/Users/storopoli/Documents/R_Scripts/Stan/8_schools.stan", String)

data = Dict(
    "J" => 8,
    "y" => [28,  8, -3,  7, -1,  1, 18, 12],
    "sigma" => [15, 10, 16, 11,  9, 11, 10, 18]
)

sm = SampleModel("8_schools", model)

fit = stan_sample(sm, data = data);

if success(fit)
    samples = read_samples(sm);
end

if success(fit)
    df = read_summary(sm, true)
    df |> display
end
