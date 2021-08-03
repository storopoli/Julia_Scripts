# Adapted from: https://gist.github.com/asinghvi17/2d192f7379c6abb1f5536be4f8f457b5

using Pkg
Pkg.activate(mktempdir())
Pkg.add([
    "CairoMakie",
    "WGLMakie",
    "ForwardDiff"
])

# using CairoMakie # No interactivity
using WGLMakie   # Interactivity
using ForwardDiff

function ascend(
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

fig = Figure()

# 2-D
# ax1 = Axis(fig[1,1])
# 3-D
ax1 = Axis3(fig[1,1])


# different plots you can see of f and ∇f
# remove comment to display

xa = LinRange(-5, 5, 500)
ya = LinRange(-5, 5, 500)
za = [f(x, y) for x ∈ xa, y ∈ ya]
# ∇za = [∇f(x, y) for x ∈ xa, y ∈ ya]

fsurf = surface!(ax1, xa, ya, za, shading = false)
# ∇fsurf = surface!(ax1, xa, ya, ∇za, shading = false)

fcont = contour!(ax1, xa, ya, za, levels = 20, linewidth = 3)
# ∇fcont = contour!(ax1, xa, ya, ∇za, levels = 20, linewidth = 3);

fheat = heatmap!(ax1, xa, ya, za);
# ∇fheat = heatmap!(ax1, xa, ya, ∇za);

fcont3 = contour3d!(ax1, xa, ya, za, levels = 20, linewidth = 3)
# ∇fcont3 = contour3d!(ax1, xa, ya, ∇za, levels = 20, linewidth = 3);

sl_x = MakieLayout.Slider(fig[2, 1], range=0.01:0.01:4, startvalue=0.1)
x0 = Node(0f0)
y0 = Node(0f0)
coords = @lift(ascend($(sl_x.value), $x0, $y0))
xs = lift(x -> x[:, 1], coords)
ys = lift(x -> x[:, 2], coords)
zs = lift((x, y) -> f.(x, y), xs, ys) # for three dimensional plots
lines!(ax1, xs, ys, zs, color = :black, linewidth = 10)

fig
