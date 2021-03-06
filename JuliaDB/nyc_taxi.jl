# see benchmarks here: https://github.com/JuliaDB/JuliaDB_Benchmarks/blob/master/bigdata/JuliaDB%20with%20TrueFX%20dataset.ipynb
using Distributed
addprocs() # Defaults to maximum cores
# or for a specific project
# after ]activate .
addprocs(exeflags="--project")

@everywhere using JuliaDB, OnlineStats, StatsPlots

# Data from: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
csvfiles = glob("*.csv", "/Users/storopoli/Desktop/nyc-taxi-data")
sum(filesize, csvfiles) / 1024^3  # size in GB

t = loadtable(
    "/Users/storopoli/Desktop/nyc-taxi-data",
    output="/Users/storopoli/desktop/bin",
    chunks=length(csvfiles),
    indexcols=[:tpep_pickup_datetime]
    )

t = load("/Users/storopoli/desktop/bin")

reduce((+, min, max), t, select=:tip_amount)
groupreduce((+, min, max), t, (:payment_type => string), select=:tip_amount)

groupreduce((Mean(), Variance()), t, (:payment_type => string), select=:tip_amount)

# Plot Stuff
partitionplot(
    t, :total_amount;
    by=(:passenger_count => string),
    stat=Hist(10),
    dropmissing=true)
