library(dplyr)

n <- 10e3
df <- tibble(x = sample(c("A", "B", "C", "D"), n, replace = TRUE),
             y = runif(n),
             z = rnorm(n))

# 3.15ms
bench::mark(
    df  %>%
    group_by(x)  %>%
    summarize(median(y),
              mean(z))
)
