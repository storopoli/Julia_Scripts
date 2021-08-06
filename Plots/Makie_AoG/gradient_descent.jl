# Adapted from: https://gist.github.com/asinghvi17/2d192f7379c6abb1f5536be4f8f457b5

using Pkg
Pkg.activate(mktempdir())
Pkg.add([
    "CairoMakie",
    "WGLMakie",
    "GLMakie",
    "ForwardDiff"
])

# using CairoMakie # No interactivity
using GLMakie   # Interactivity
using ForwardDiff

function descend(
    Δ::Real,
    x0::Real,
    y0::Real;
    numsteps = 10
    )::Array{Float64, 2}

    coords = zeros(numsteps, 2)

    coords[1, 1] = x0

    coords[1, 2] = y0

    for i ∈ 2:numsteps

        coords[i, :] = coords[i-1, :] + Δ*∇f(coords[i-1, :])

        end

    coords
    end

f(x::Real, y::Real)  = 2/(x^2 - 4*x + y^2 + 5) + 3/(x^2 - 4*y + y^2 + 6)

f(x::Array{T, 1} where T <: Real)  = f(x[1], x[2])

∇f(x::Real, y::Real) = ForwardDiff.gradient(f, [x, y])
∇f(x::Array{<:AbstractFloat, 1}) = ForwardDiff.gradient(f, x)

fig = Figure(resolution=(800, 800))

# 2-D
ax1 = Axis(fig[1, 1])
# 3-D
ax2 = Axis3(fig[2, 1])

# different plots you can see of f and ∇f
xa = LinRange(-5, 5, 500)
ya = LinRange(-5, 5, 500)
za = [-f(x, y) for x ∈ xa, y ∈ ya]
# ∇za = [∇f(x, y) for x ∈ xa, y ∈ ya]

plotobj2d = contour3d!(ax1, xa, ya, za; shading=false, linewidth=3, levels=20)
plotobj3d = surface!(ax2, xa, ya, za; shading=false)
# fsurf = surface!(ax1, xa, ya, za; shading = false)
# ∇fsurf = surface!(ax1, xa, ya, ∇za; shading = false)
# fcont = contour!(ax1, xa, ya, za; levels = 20, linewidth = 3)
# ∇fcont = contour!(ax1, xa, ya, ∇za; levels = 20, linewidth = 3)
# fheat = heatmap!(ax1, xa, ya, za)
# ∇fheat = heatmap!(ax1, xa, ya, ∇za)
# fcont3 = contour3d!(ax1, xa, ya, za; levels = 20, linewidth = 3)
# ∇fcont3 = contour3d!(ax1, xa, ya, ∇za; levels = 20, linewidth = 3)

slider_η = labelslider!(
    fig,
    "η",
    0.01:0.001:2.0;
    width = 350,
    tellwidth=false)

# slider_iter = labelslider!(
#     fig,
#     "Iterações",
#     10:10:100)

fig[3, 1]= slider_η.layout
# fig[4, 1]= slider_iter.layout
x0 = Node(0.0)
y0 = Node(0.0)
coords = @lift(descend($(slider_η.slider.value), $x0, $y0))
xs = lift(x -> x[:, 1], coords)
ys = lift(x -> x[:, 2], coords)
zs = lift((x, y) -> .-f.(x, y), xs, ys) # for three dimensional plots
scatterlines!(ax1, xs, ys, zs, color = :red, linewidth = 5)
scatterlines!(ax2, xs, ys, zs, color = :red, linewidth = 5)
hidedecorations!(ax1)
hidedecorations!(ax2)
hidespines!(ax1)
hidespines!(ax2)

fig
