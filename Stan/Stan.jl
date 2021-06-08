using Stan
using StanSample
using DataFrames
using MCMCChains

model_file = "/Users/storopoli/Documents/R_Scripts/Stan/8_schools.stan"
model = reduce(*, readlines(model_file))

# Create and compile a SampleModel object:
sm = SampleModel("8_schools", model, tmpdir=pwd())

# The observed input data as a Dict:
data = Dict(
    "J" => 8,
	"y" => [28,  8, -3,  7, -1,  1, 18, 12],
    "sigma" => [15, 10, 16, 11,  9, 11, 10, 18]
)

df = DataFrame(data)

# The observed input data as a NamedTuple:

data = (
    J = nrow(df),
    y = df.y,
    sigma = df.sigma
)

# Run a simulation by calling stan_sample(), passing in the model and data:
stan_sample(sm; data)

samples_nt = read_samples(sm) # Default
samples_df = read_samples(sm; output_format=:dataframes)
chain = read_samples(sm; output_format=:mcmcchains, include_internals=true, return_parameters=true)
