using CmdStan, MCMCChains, StatsPlots
using DataFrames, Pipe, CSV, HTTP
using Random:seed!
plotly() # only workks with Plotly (don't ask me why), maybe because there is a lot of parameters to Plot in this particular model

seed!(1)

df = @pipe HTTP.get("https://github.com/selva86/datasets/blob/master/mpg_ggplot2.csv?raw=TRUE").body |>
    CSV.read(_, DataFrame)

#### Data Prep ####
idx_map = Dict(key => idx for (idx, key) in enumerate(unique(df.class)))
y = df[:, :hwy]
idx = getindex.(Ref(idx_map), df.class)
X = Matrix(select(df, [:displ, :year])) # the model matrix

dat = Dict(
    "N" => size(X, 1),
    "J" => length(unique(idx)),
    "K" => size(X, 2),
    "id" => idx,
    "X" => X,
    "y" => y
)

model =
"""
data {
  int<lower=1> N; //the number of observations
  int<lower=1> J; //the number of groups
  int<lower=1> K; //number of columns in the model matrix
  int<lower=1,upper=J> id[N]; //vector of group indeces
  matrix[N,K] X; //the model matrix
  vector[N] y; //the response variable
}
transformed data {
  matrix[N, K] Q_ast;
  matrix[K, K] R_ast;
  matrix[K, K] R_ast_inverse;

  // thin and scale the QR decomposition
  Q_ast = qr_thin_Q(X) * sqrt(N - 1);
  R_ast = qr_thin_R(X) / sqrt(N - 1);
  R_ast_inverse = inverse(R_ast);
}
parameters {
  real alpha; // population-level intercept
  vector[K] theta; // coefficients on Q_ast
  //vector[J] varying_alpha; // group-level regression intercepts
  vector[J] z_j; // non centered group-level regression intercepts
  real<lower=0> sigma; // model residual error
  real<lower=0> sigma_alpha; //standard error for the group-level regression intercepts
}
model {
  //priors
  alpha ~ normal(mean(y), 2.5 * sd(y));
  theta ~ student_t(3, 0, 1);
  //varying_alpha ~ normal(0, sigma_alpha);
  z_j ~ normal(0, 1);
  sigma ~ exponential(1 / sd(y));
  sigma_alpha ~ cauchy(0, 2.5);

  //likelihood
  y ~ normal(alpha + z_j[id] * sigma_alpha + Q_ast * theta, sigma);
}
generated quantities {
  vector[K] beta; // reconstructed population-level regression coefficients
  beta = R_ast_inverse * theta; // coefficients on X
  vector[J] varying_alpha; // reconstructed group-level regression intercepts
  varying_alpha = z_j * sigma_alpha; // varying intercepts
  real y_rep[N] = normal_rng(alpha + z_j[id] * sigma_alpha + X * beta, sigma);
}
"""

stanmodel = Stanmodel(Sample(), name="mpg_ncp", model=model, output_format=:mcmcchains)
rc, chns, cnames = stan(stanmodel, dat)

p1 = plot(chns)
p2 = pooleddensity(chns)

MCMCChains.traceplot(chns)
traceplot(read_summary(rc), )
sdf = read_summary(stanmodel)
