library(cmdstanr)

set.seed(1)

nd <- 5
ns <- 10
n <- nd * ns
a0 <- 1
a1 <- 0.5
a0_sig <- 0.3

y <- rep(0, n)
x <- rep(0, n)
idx <- rep(0, n)
i <- 0
for (s in 1:ns) {
    a0s <- rnorm(1, 0, 0.3)
    logpop <- rnorm(1, 9, 1.5)
    lambda <- exp(a0 + a0s + a1 * logpop)
    for (nd in 1:nd) {
        i <- i + 1
        x[i] <- logpop
        idx[i] <- s
        y[i] <- rpois(1, lambda)
    }
}

data <- list(
    y = y,
    x = x,
    idx = idx,
    N = n,
    Ns = ns
)

# 19.2s (sampling 3.8s)
system.time({
    model <- cmdstan_model(here::here("Turing", "Benchmarks", "h_poisson.stan"))
    fit <- model$sample(data = data)
})

# ESS a0_sig 943
fit$cmdstan_summary()
