library(cmdstanr)

set.seed(1)
data <- list(
    J = 8,
    y = c(28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0),
    sigma = c(15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0)
)

# 12s (sampling 0.5s)
system.time({
    model <- cmdstan_model(here::here("Turing", "Benchmarks", "8_schools.stan"))
    fit <- model$sample(data = data)
})

# ESS theta[1] 1082
fit$cmdstan_summary()
