using Gadfly

x = [cos(t) for t in LinRange(0, 2pi, 20)]
y = [sin(t) for t in LinRange(0, 2pi, 20)]
colors = 1:20

# 27s
@time plot(
    [x y],
    x = Col.value(1),
    y = Col.value(2),
    Geom.point,
    color = colors,
    size = [15],
)
#### Reset!
using Plots

x = [cos(t) for t in LinRange(0, 2pi, 20)]
y = [sin(t) for t in LinRange(0, 2pi, 20)]
colors = 1:20

# 3.5s 355MB alloc
@time scatter(x, y, markercolor = colors, markersize = 15)
