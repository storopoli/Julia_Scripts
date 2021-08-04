# Beta version... it works, a lot of things could be better.
using CairoMakie, LinearAlgebra, Random, ColorSchemes
using Downloads: download
using FileIO, Dates
import GeometryBasics
CairoMakie.activate!()

# https://discourse.julialang.org/t/how-to-see-all-installed-packages-a-few-other-pkg-related-questions/1231/15
using Pkg
versions = []
for s in ["Makie", "CairoMakie"]
    io = IOBuffer()
    Pkg.status([s]; io=io)
    textPkg = String(take!(io))
    push!(versions, split(textPkg, '\n')[2:end-1][1][14:end])
end


function makiePlottingFunctions()
    fig = Figure(resolution = (950,1250), fontsize = 14, font = "CMU Serif")
    ax1 = Axis(fig[1,1], title = "lines(x,y)")
    ax2 = Axis3(fig[1,2], title = "lines(x,y,z)",
        xlabel = "", ylabel = "", zlabel = "")
    ax3 = Axis(fig[1,3], title = "scatter(x,y)")
    ax4 = Axis(fig[1,5], title = "meshscatter(x,y)")
    ax5_1 = Axis3(fig[1,4], title = "scatter(x,y,z)",
        xlabel = "", ylabel = "", zlabel = "")

    ax5 = Axis(fig[2,1], title = "scatterlines(x,y)")
    ax6 = Axis3(fig[2,2], title = "scatterlines(x,y,z)",
        xlabel = "", ylabel = "", zlabel = "")
    ax7 = Axis(fig[2,3], title = "arrows(x,y,u,v)")
    ax8 = Axis(fig[2,4], title = "streamplot(f,-i..i, -j..j)")
    ax9 = Axis(fig[2,5], title = "stem(x,y)")

    ax10 = Axis(fig[3,1], title = "series(curves)")
    ax11 = Axis(fig[3,2], title = "stairs(x,y)")
    ax12 = Axis(fig[3,3], title = "linesegments(x,y)")
    ax13 = Axis(fig[3,4], title = "rangebars(vals,lo,hi)")
    ax14 = Axis(fig[3,5], title = "errorbars(x,y,lo,hi)")

    ax15 = Axis(fig[4,1], title = "crossbar(x,y,lo,hi)")
    ax16 = Axis(fig[4,2], title = "boxplot(x,y)")
    ax17 = Axis(fig[4,3], title = "barplot(x,y)")
    ax18 = Axis(fig[4,4], title = "hist(x)")
    ax19 = Axis(fig[4,5], title = "density(x)")

    ax20 = Axis(fig[5,1], title = "violin(x,y)")
    ax21 = Axis(fig[5,2], title = "band(x,lo,hi)")
    ax22 = Axis(fig[5,3], title = "pie(fracs)")
    ax23 = Axis(fig[5,4], title = "poly(points)")
    ax24 = Axis(fig[5,5], title = "text(s)")

    ax25 = Axis(fig[6,1], title = "image(img)")
    ax26 = Axis(fig[6,2], title = "heatmap(x,y,vals)")
    ax27 = Axis(fig[6,3], title = "contour(x,y,vals)")
    ax28 = Axis(fig[6,4], title = "contourf(x,y,vals)")
    ax29 = Axis(fig[6,5], title = "mesh(v,f)")

    supertitle = fig[0, :] = Label(fig, "Plotting Functions with Makie :: CHEAT SHEET",
    textsize = 28, color = :black)
    supertitle.padding = (0, 6, 16, 0)

    makieRef = "http://makie.juliaplots.org"
    bookRef = "Learn more in Julia Data Science. https://juliadatascience.io"
    footnote = fig[end+1, :] = Label(fig,
        bookRef * "  ⋅  " * makieRef * "  ⋅ $(versions[1]) ⋅ $(versions[2]) ⋅ Updated: $(today())",
    textsize = 12, color = :black, halign = :right)

    axMakieLogo = Axis(fig, bbox = BBox(10, 100, 1150, 1250),
        backgroundcolor=:transparent)
    hidedecorations!(axMakieLogo)
    hidespines!(axMakieLogo)

    logoMakie = load(joinpath(pwd(), "Plots", "Makie_AoG", "makie_logo_white.png"))
    image!(axMakieLogo, rotr90(logoMakie))
    # The inset axis
    #inset_ax = Axis(fig[0, end],
    #    width=Relative(0.5),
    #    height=Relative(0.5),
    #    halign=0.0,
    #    valign=0.0,
    #    backgroundcolor=:lightgray)


    axs = [ax1, ax2, ax3, ax4, ax5, ax5_1, ax6, ax7, ax8, ax9,
        ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17,
        ax18, ax19, ax20, ax21, ax22, ax23, ax24, ax25,
        ax26, ax27, ax28, ax29]
    x = -1.5π:0.5:4

    xs = LinRange(-3, 3, 10)
    ys = LinRange(-3, 3, 10)
    us = [x + y for x in xs, y in ys]
    vs = [y - x for x in xs, y in ys]
    strength = vec(sqrt.(us .^2 .+ vs .^2))

    lines!(ax1, x, sin.(x))

    lines!(ax2, x, sin.(x), cos.(x), color = :red)
    ylims!(ax2, -0.8, 0.8)

    scatter!(ax3, x, sin.(x), markersize = 5, color = :black)

    meshscatter!(ax4, x, sin.(x), cos.(x), markersize = 0.1)

    scatterlines!(ax5, x, sin.(x), markersize = 5)

    scatter!(ax5_1, x, sin.(x), cos.(x), markersize = 5)
    ylims!(ax5_1, -0.8, 0.8)

    scatterlines!(ax6, x, sin.(x), cos.(x), markersize = 5)
    ylims!(ax6, -0.8, 0.8)

    arrows!(ax7, xs,ys,us,vs, arrowsize = 5, lengthscale = 0.1,
            arrowcolor = strength, linecolor = strength,
            colormap = :plasma)
    limits!(ax7, -3,3,-3,3)

    semiStable(x,y) = Point(-y+ x*(-1+x^2+y^2)^2, x+y*(-1+x^2+y^2)^2)

    streamplot!(ax8, semiStable, -4..4, -4..4, colormap = :plasma,
                gridsize= (20,20), arrow_size = 7,linewidth=1)

    stem!(ax9, x, exp.(-x/5).*cos.(x), color = x,
        stemcolor = x, markersize = 5)

        Random.seed!(13)
    series!(ax10, cumsum(randn(4, 10),dims =2), markersize=5,
        color=:Set1)
    stairs!(ax11, x, exp.(-x.^2/2); step=:center, color=:black, linestyle=nothing)

    linesegments!(ax12, cos.(x), x, linewidth = 3,
        color = 1:length(x), colormap = :CMRmap)

    vals = -1:0.5:1
    lows = zeros(length(vals)) .+ rand(length(vals))
    highs = LinRange(0.1, 0.4, length(vals)) .+ 2rand(length(vals))

    rangebars!(ax13, vals, lows, highs, color = LinRange(0, 1, length(vals)),
        whiskerwidth = 10, direction = :y, colormap = :Set1_5)

    xs = 0:10
    ys = rand(length(xs)) .+ sin.(xs)

    lowerrors = fill(0.1, length(xs))
    higherrors = LinRange(0.1, 0.3, length(xs))

    errorbars!(ax14, xs, ys, lowerrors, higherrors,
        direction = :y, whiskerwidth = 10)
    #errorbars!(ax14, xs, 2*ys, lowerrors, 5higherrors,
    #    direction = :x, color = :red, whiskerwidth = 7)
    xs = 1:4
    ys = rand(4)
    ymins = ys .- 1
    ymaxs = ys .+ 1

    crossbar!(ax15, xs, ys, ymins, ymaxs, show_notch = true,
        color = 1:4, colormap = :Set1_4)

    xs = rand(1:3, 1000)
    ys = randn(1000)

    boxplot!(ax16, xs, ys, orientation = :vertical,
        color = xs, outliercolor = :black, whiskerwidth=0.5,
        colormap = (:grey, :orange))

    #p = Makie.LinePattern(width=10, tilesize=(30,30),
    #    linecolor=:orange, background_color=:green);
    barplot!(ax17, [1,2,3], [2,1.5,2.5], strokewidth=2,
        color=[:grey90,:black, (:orange, 0.5)],)
    #barplot!(ax17, [3], [2.5], color = p )
    hist!(ax18, randn(1000), bins = 9, color = :values)

    density!(ax19, randn(1000), color = "grey")
    density!(ax19, 3 .+ randn(1000), color = (:orange,0.2),
        strokewidth = 1, linestyle = :dash, strokecolor = :red)

    for i in 1:3
        violin!(ax20, fill(i,1000), randn(1000), show_median=true,
            strokewidth = 0.5, mediancolor = :black)
    end

    x = y = -10:0.11:10
    y1d =  sin.(x) ./ x
    lower = y1d .- 0.2
    upper = y1d .+ 0.2
    band!(ax21, x, lower, upper; color = (:red, 0.5))

    fracs = [0.1, 0.3, 0.4, 0.2]
    cbarPal = :sunset
    cmap = get(colorschemes[cbarPal], LinRange(0,1,length(fracs)))

    pie!(ax22, fracs, color = cmap)


    poly!(ax23, Point2f0[(0, 0), (1, 1), (2, 0), (1,-1)], color = :grey90,
        strokecolor = :black, strokewidth = 1)

    text!(ax24, "Say something \n funny 😅", position = (0,1),
        textsize = 12, align = (:center, :center),
        rotation = π/4, color = "#000080")

    url = "https://testimages.juliaimages.org/images/monarch_color_256.png"
    img = load(download(url))
    image!(ax25, img)

    heatmap!(ax26, 1:10, 1:10, rand(10,10))

    contour!(ax27, 1:10, 1:10, rand(10,10))
    contourf!(ax28, 1:10, 1:10, rand(10,10), colormap = :plasma)

    vertices = [
        0.0 0.0;
        1.0 0.0;
        1.0 1.0;
        0.0 1.0;
    ]

    faces = [
        1 2 3;
        3 4 1;
    ]
    cbarPal = :gist_rainbow
    cmap = get(colorschemes[cbarPal], LinRange(0,1,length(fracs)))

    mesh!(ax29, vertices, faces, color = cmap, shading = false)

    [hidedecorations!(axs[i], grid=false) for i in 1:length(axs)]
    colgap!(fig.layout, 0)
    #rowgap!(fig.layout, 10)
    fig
end
fig = with_theme(theme_light(),
    Axis = (; titlecolor = :black, aspect = AxisAspect(1)),
    Axis3 = (; titlecolor = :black, aspect = (1,1,1))) do
    makiePlottingFunctions()
end
#fig
save("makiePlottingFunctionsHide.png", fig, px_per_unit = 2.0)

using GLMakie
GLMakie.activate!()

versions = []
for s in ["Makie", "GLMakie"]
    io = IOBuffer()
    Pkg.status([s]; io=io)
    textPkg = String(take!(io))
    push!(versions, split(textPkg, '\n')[3:end-1][1][14:end])
end


function GLMakiePlottingFunctions()
    Random.seed!(123)
    fig = Figure(resolution = (2*900, 2*3.7*1200 ÷ 6), font = "CMU Serif")
    ax1 = Axis3(fig[1,1], title = "wireframe(x,y,z)",
        perspectiveness = 0.5, aspect = (1,1,1),
        xlabel = "", ylabel = "", zlabel = "")
    ax2 = Axis3(fig[1,2], title = "surface(x,y,z)",
        perspectiveness = 0.5, aspect = (1,1,1),
        xlabel = "", ylabel = "", zlabel = "")
    ax3 = Axis3(fig[1,3], title = "contour3d(x,y,z)",
        perspectiveness = 0.5, aspect = (1,1,1),
        xlabel = "", ylabel = "", zlabel = "")
    ax4 = Axis3(fig[1,4], title = "volume(x,y,z,vals)",
        perspectiveness = 0.5, aspect = (1,1,1),
        xlabel = "", ylabel = "", zlabel = "")
    ax5 = Axis3(fig[1,5], title = "heatmap(x,y,vals,\ntransformation=(:xy,0.5)",
        perspectiveness = 0.5, aspect = (1,1,1),
        xlabel = "", ylabel = "", zlabel = "")
    ax6 = Axis3(fig[2,1], title = "meshscatter(x,y,z)",
        perspectiveness = 0.5, aspect = (1,1,1),
        xlabel = "", ylabel = "", zlabel = "")
    ax7 = Axis3(fig[2,2], title = "arrows(x,y,u,v,\n transformation=(:yz,-2))",
        perspectiveness = 0.5, aspect = (1,1,1),
        xlabel = "", ylabel = "", zlabel = "")
    ax8 = Axis3(fig[2,3], title = "arrows(points,directions)",
        perspectiveness = 0.5, aspect = (1,1,1),
        xlabel = "", ylabel = "", zlabel = "")

    ax9 = Axis3(fig[2,4], title = "streamplot(f,-i..i,-j..j,\ntransformation=(:xy,-1))",
        perspectiveness = 0.5, aspect = (1,1,1),
        xlabel = "", ylabel = "", zlabel = "")
    ax10 = Axis3(fig[2,5], title = "streamplot(f,-i..i,-j..j,-k..k)",
        perspectiveness = 0.5, aspect = (1,1,1),
        xlabel = "", ylabel = "", zlabel = "")

    ax11 = Axis3(fig[3,1], title = "stem(x,y,z)",
        perspectiveness = 0.5, aspect = (1,1,1),
        xlabel = "", ylabel = "", zlabel = "")
    ax12 = Axis3(fig[3,2], title = "linesegments(x,y,z)",
        perspectiveness = 0.5, aspect = (1,1,1),
        xlabel = "", ylabel = "", zlabel = "")

    ax13 = Axis3(fig[3,3], title = "band(lo,hi)",
        perspectiveness = 0.5, aspect = (1,1,1),
        xlabel = "", ylabel = "", zlabel = "")

    ax14 = Axis3(fig[3,4], title = "contour(x,y,z,vals)",
        perspectiveness = 0.5, aspect = (1,1,1),
        xlabel = "", ylabel = "", zlabel = "")
    ax15 = Axis3(fig[3,5], title = "mesh(obj)",
        perspectiveness = 0.5, aspect = (1,1,1),
        xlabel = "", ylabel = "", zlabel = "")

    axs = [ax1, ax2, ax3, ax4, ax5, ax5_1, ax6, ax7, ax8, ax9,
        ax10, ax11, ax12, ax13, ax14, ax15]


    supertitle = fig[0, :] = Label(fig, "Plotting Functions with Makie :: CHEAT SHEET",
    textsize = 45, color = :black)
    supertitle.padding = (0, 12, 32, 0)

    makieRef = "http://makie.juliaplots.org"
    bookRef = "Learn more in Julia Data Science. https://juliadatascience.io"
    footnote = fig[end+1, :] = Label(fig,
        bookRef * "  ⋅  " * makieRef * "  ⋅ $(versions[1]) ⋅ $(versions[2]) ⋅ Updated: $(today())",
    textsize = 22, color = :black, halign = :right)

    axMakieLogo = Axis(fig, bbox = BBox(15, 190, 1305, 1480),
        backgroundcolor=:transparent)
    hidedecorations!(axMakieLogo)
    hidespines!(axMakieLogo)

    logoMakie = load(joinpath(pwd(), "Plots", "Makie_AoG", "makie_logo_white.png"))
    image!(axMakieLogo, rotr90(logoMakie))


    x = y =  LinRange(-2, 2, 15)
    z = (-x .* exp.(-x .^ 2 .- (y') .^ 2)) .* 4
    wireframe!(ax1, x, y, z)
    surface!(ax2, x, y, z)
    contour3d!(ax3, x, y, z, levels = 15,)
    xv = 1:10
    yv = 1:10
    zv = 1:10
    vol = randn(10,10,10)
    volume!(ax4, xv, yv, zv, vol, colormap = Reverse(:plasma))
    heatmap!(ax5, 1:10, 1:10, randn(10,10),
        transformation=(:xy, + 0.5), colormap = Reverse(:plasma))
    meshscatter!(ax6, rand(6), rand(6), rand(6), color = 1:15, markersize =0.15,
        marker = FRect3D(Vec3f0(0), Vec3f0(1)), colormap=:gnuplot)

    xs = LinRange(-3, 3, 10)
    ys = LinRange(-3, 3, 10)
    us = [x + y for x in xs, y in ys]
    vs = [y - x for x in xs, y in ys]
    strength = vec(sqrt.(us .^2 .+ vs .^2))

    arrows!(ax7, xs,ys,us,vs, arrowsize = 15, lengthscale = 0.1,
            arrowcolor = strength, linecolor = strength,
            colormap = :inferno, transformation= (:yz,-2))
    zlims!(ax7,-3,3)

    # http://makie.juliaplots.org/stable/plotting_functions/arrows.html
    ps = [Point3f0(x, y, z) for x in -3:1:3 for y in -3:1:3 for z in -3:1:3]
    ns = map(p -> 0.1*rand() * Vec3f0(p[2], p[3], p[1]), ps)
    lengths = norm.(ns)
    arrows!(ax8, ps, ns, fxaa=true, # turn on anti-aliasing
        color=lengths,
        linewidth = 0.1, arrowsize = Vec3f0(0.3, 0.3, 0.4),
        align = :center)

    semiStable(x,y) = Point(-y+ x*(-1+x^2+y^2)^2, x+y*(-1+x^2+y^2)^2)

    streamplot!(ax9, semiStable, -4..4, -4..4, colormap = :plasma,
        gridsize= (20,20), arrow_size = 0.25,linewidth=1,
        transformation= (:xy,-1))

    flowField(x,y,z) = Point(-y+ x*(-1+x^2+y^2)^2, x+y*(-1+x^2+y^2)^2, z + x*(y-z^2))

    streamplot!(ax10, flowField, -4..4, -4..4, -4..4, colormap = :plasma,
        gridsize= (7,7), arrow_size = 0.25,linewidth=1)

    xs = LinRange(-pi, 2pi, 15)
    stem!(ax11, 0.5xs, -sin.(xs), cos.(xs), offset = Point3f0.(0.5xs, sin.(xs)/2, cos.(xs)/2),
        stemcolor = 1:length(xs),)

    linesegments!(ax12, cos.(xs), xs, sin.(xs), linewidth = 5,
            color = 1:length(xs), colormap = :plasma)


            lower = [Point3f0(i,-i,0) for i in LinRange(0,3,100)]
    upper = [Point3f0(i,-i, sin(i) * exp(-(i+i)) ) for i in range(0,3, length=100)]

    band!(ax13, lower, upper, color=repeat(norm.(upper), outer=2), colormap = :CMRmap)

    contour!(ax14, 1:10, 1:10, 1:10, rand(10, 10,10), levels = 3,
        colormap = :thermal)

    #sphere = Sphere(Point3f0(0), 1)
    #spheremesh = GeometryBasics.mesh(sphere)
    rectMesh = FRect3D(Vec3f0(-0.5), Vec3f0(1))
    recmesh = GeometryBasics.mesh(rectMesh)
    colors = [rand() for v in recmesh.position]
    mesh!(ax15, recmesh, color= colors, colormap = :rainbow, shading = false)
    #rowgap!(fig.layout, 5)
    #colgap!(fig.layout, 5)
    [hidedecorations!(axs[i], grid=false) for i in 1:length(axs)]
    fig
end
#GLMakiePlottingFunctions()

fig = with_theme(theme_light(), figure_padding = (20,15,10,60),
    fontsize = 25, Axis3 = (; titlesize=27, titlegap=0,titlecolor = :black)) do
    GLMakiePlottingFunctions()
end
#rowgap!(fig.layout, 15)
#colgap!(fig.layout, 5)
save("GLMakiePlottingFunctionsHide.png", fig)
