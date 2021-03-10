# brew install awscli
# aws s3 cp s3://ursa-labs-taxi-data ursa-labs-taxi-data --recursive
# 2009 to 2015 (32GB)
# Initial RAM usage 16GB and stayed there for the whole time!

using Parquet, DataFrames, Tables, Chain, Query
using BenchmarkTools, Statistics

dir = "/Users/storopoli/Desktop/ursa-labs-taxi-data/"
tbl = read_parquet(dir)
df = DataFrame(tbl)
Tables.partitions(tbl)

keys(tbl)

temp = Tables.datavaluerows(tbl) |>
		@groupby(_.payment_type) |>
        @map({
            payment_type = key(_),
            nrow = length(_),
            mean_total_amount = mean(_.tip_amount)
        }) |>
        DataFrame
