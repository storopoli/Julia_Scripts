# using GLMakie
# using WGLMakie
using CairoMakie

function rosenbrock(x::T, y::T) where T <: AbstractFloat
  return (1.0 - x)^2 + 100.0 * (y - x^2)^2
end

xs, ys = LinRange(-1.5, 1.5, 1000), LinRange(-1.5, 1.5, 1000)
zs = [rosenbrock(xᵢ,yᵢ) for xᵢ ∈ xs, yᵢ ∈ ys]

fig = Figure()
ax1, s = surface(fig[1, 1], xs, ys, zs;
                    axis=(; type=Axis3, azimuth=2.25pi))
ax2, c = contourf(fig[1, 2], xs, ys, zs)
cbar = fig[1, end+1] = Colorbar(fig, c)
supertitle = fig[0, :] = Label(fig, "Rosenbrock", textsize = 36)
hidedecorations!(ax1)
hidedecorations!(ax2)
hidespines!(ax1)
hidespines!(ax2)
fig
