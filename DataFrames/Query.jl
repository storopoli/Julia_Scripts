using DataFrames, Query, RDatasets, Statistics

mtcars = dataset("datasets", "mtcars")

mtcars |> @map(mean)
