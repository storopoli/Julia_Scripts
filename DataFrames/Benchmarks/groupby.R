library(dplyr)
library(data.table)

n <- 10e3
df <- tibble(
    x = sample(c("A", "B", "C", "D"), n, replace = TRUE),
    y = runif(n),
    z = rnorm(n)
)

dt <- as.data.table(df)

# dplyr 3.17ms (1.42ms M1) (3.11ms Dell G5)
# data.table 697µs (776µs M1)
bench::mark(
    dplyr = df %>%
        group_by(x) %>%
        summarize(
            median(y),
            mean(z)
        ),
    data.table = dt[, .(mean(y), mean(z)), x],
    check = FALSE
)
