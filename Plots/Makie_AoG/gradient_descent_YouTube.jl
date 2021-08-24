# Adapted from: https://gist.github.com/asinghvi17/2d192f7379c6abb1f5536be4f8f457b5
# Thanks @lazarusA

using Pkg
Pkg.activate(mktempdir())
Pkg.add([
    "GLMakie",
    "ForwardDiff"
])

using GLMakie   # Interactivity
using ForwardDiff

function descend( Δ::Real, x0::Real, y0::Real; numsteps = 10)::Array{Float64, 2}
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

with_theme(theme_dark()) do
    fig = Figure(resolution=(800, 800))

    # 3-D

    ax1 = Axis3(fig[1, 1], aspect = (0.5,0.5,0.5), viewmode = :fit,
        perspectiveness = 0.5, elevation = pi/10)

    xa = LinRange(-5, 5, 500)
    ya = LinRange(-5, 5, 500)
    za = [-f(x, y) for x ∈ xa, y ∈ ya]

    plotobj3d = surface!(ax1, xa, ya, za; shading=false,
        colormap = (:CMRmap, 0.8), transparency = false)

    coords = descend(0.6, -1, -1.5; numsteps=20)
    xs = coords[:, 1]
    ys = coords[:, 2]
    zs = .-f.(xs, ys) # for three dimensional plots
    scatterlines!(ax1, xs, ys, zs, linewidth = 5, color = :red)

    Label(fig[2,1], "Aula de Hoje no Youtube: youtube.com/uninove", textsize = 32, color=:red)
    Label(fig[3,1], "Feito com Julia e Makie.jl, veja mais em juliadatascience.io", textsize = 24, color=:orange)
    Label(fig[0, :], "Gradient Descent", textsize = 28)

    hidedecorations!(ax1)
    hidespines!(ax1)
    rowgap!(fig.layout, 12)
    fig
    record(fig, "descend.mp4") do io
        for i in -2π:0.015:2π
            ax1.azimuth = i
            recordframe!(io)
        end
    end
end
