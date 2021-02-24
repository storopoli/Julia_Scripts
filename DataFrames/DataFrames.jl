using DataFrames, Pipe, StatsPlots
using Statistics: mean, median
using RDatasets

mtcars = dataset("datasets", "mtcars")

@pipe mtcars |>
    groupby(_, [:AM, :VS]) |>
    combine(_, [:HP, :MPG] .=> mean)

@pipe mtcars |>
    filter(:HP => >(100), _) |>
    sort(_, :HP, rev=true)
