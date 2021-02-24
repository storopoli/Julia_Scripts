using WGLMakie

points = [Point2f0(cos(t), sin(t)) for t in LinRange(0, 2pi, 20)]
colors = 1:20
figure, axis, scatterobject = scatter(points, color = colors, markersize = 15)
figure

# 31s 3.76GB alloc
@time scatter(points, color = colors, markersize = 15)

#### Reset!
using Plots

x = [cos(t) for t in LinRange(0, 2pi, 20)]
y = [sin(t) for t in LinRange(0, 2pi, 20)]
colors = 1:20

# 3.5s 355MB alloc
@time scatter(x, y, markercolor = colors, markersize = 15)
