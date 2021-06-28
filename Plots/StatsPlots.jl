using CSV
using DataFrames
using HTTP
using StatsPlots

url = "https://cdn.jsdelivr.net/gh/allisonhorst/palmerpenguins@433439c8b013eff3d36c847bb7a27fa0d7e353d8/inst/extdata/penguins.csv"

penguins = dropmissing(CSV.read(HTTP.get(url).body, DataFrame))
@df penguins scatter(:bill_length_mm, :bill_depth_mm, group=:species)
