# http://makie.juliaplots.org/stable/makielayout/axis.html#Log-scales-and-other-axis-scales

using CairoMakie
using Makie
using AlgebraOfGraphics

f = Figure(resolution = (800, 700))

lines(f[1, 1], -100:0.1:100, axis = (
    yscale = Makie.pseudolog10,
    title = "Pseudolog scale",
    yticks = [-100, -10, -1, 0, 1, 10, 100]))

lines(f[2, 1], -100:0.1:100, axis = (
    yscale = Makie.Symlog10(10.0),
    title = "Symlog10 with linear scaling between -10 and 10",
    yticks = [-100, -10, 0, 10, 100]))

f
