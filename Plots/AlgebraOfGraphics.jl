using AlgebraOfGraphics
using CairoMakie
using CSV
using DataFrames
using HTTP

url = "https://cdn.jsdelivr.net/gh/allisonhorst/palmerpenguins@433439c8b013eff3d36c847bb7a27fa0d7e353d8/inst/extdata/penguins.csv"

penguins = dropmissing(CSV.read(HTTP.get(url).body, DataFrame; missingstring="NA"))

axis = (width = 225, height = 225)

penguin_bill = data(penguins) * mapping(
    :bill_length_mm => (t -> t / 10) => "bill length (cm)",
    :bill_depth_mm => (t -> t / 10) => "bill depth (cm)",
)

layers = linear() + mapping(marker=:sex)
plt = penguin_bill * layers * mapping(color=:species)
draw(plt)
