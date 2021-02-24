# https://www.appsloveworld.com/sample-csv-file/
using Arrow, Tables, CSV, Pipe
using BenchmarkTools, Query, Tables, DataFrames, Statistics
Threads.nthreads()

# 20k Records
download(
    "https://docs.google.com/uc?export=download&id=1a0RGUW2oYxyhIQYuezG_u8cxgUaAQtZw",
    "Arrow/df.csv",
)


df = CSV.File("Arrow/df.csv") |> DataFrame
describe(df)

# 6.6ms
@benchmark df |>
           @groupby(_.var"Company Name") |>
           @map({
               Company = _.var"Company Name",
               Leave_mean = mean(_.Leave),
               Leave_median = median(_.Leave),
           }) |>
           DataFrame

Arrow.write("Arrow/df.arrow", df)
tbl = Arrow.Table("Arrow/df.arrow")

# 19ms
@benchmark Tables.datavaluerows(tbl) |>
           @groupby(_.var"Company Name") |>
           @map({
               Company = _.var"Company Name",
               Leave_mean = mean(_.Leave),
               Leave_median = median(_.Leave),
           }) |>
           DataFrame


## Everything Now
# 6.5ms
@benchmark @pipe DataFrame(CSV.File("Arrow/df.csv")) |>
                 groupby(_, :"Company Name") |>
                 combine(_, :Leave => mean, :Leave => median)

# 10ms
@benchmark DataFrame(CSV.File("Arrow/df.csv")) |>
           @groupby(_.var"Company Name") |>
           @map({
               Company = key(_),
               Leave_mean = mean(_.Leave),
               Leave_median = median(_.Leave),
           }) |>
           DataFrame

# 19ms
@benchmark Tables.datavaluerows(Arrow.Table("Arrow/df.arrow")) |>
           @groupby(_.var"Company Name") |>
           @map({
               Company = key(_),
               Leave_mean = mean(_.Leave),
               Leave_median = median(_.Leave),
           }) |>
           DataFrame
