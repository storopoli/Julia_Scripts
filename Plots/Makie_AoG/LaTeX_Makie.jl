# https://discourse.julialang.org/t/ann-makie-jl-0-15/64704

using CairoMakie
using AlgebraOfGraphics
using DataFrames

# CairoMakie
f = Figure()
ax = Axis(f[1, 1])

lines!(0..10, x -> sin(3x) / (cos(x) + 2),
    label = L"\frac{\sin(3x)}{\cos(x) + 2}")
lines!(0..10, x -> sin(x^2) / (cos(sqrt(x)) + 2),
    label = L"\frac{\sin(x^2)}{\cos(\sqrt{x}) + 2}")

Legend(f[1, 2], ax)

f

# GLMakie
using GLMakie
lines(0..25, x -> 4 * sin(x) / (cos(3x) + 4), figure = (fontsize = 25,),
    axis = (
        xticks = (0:10:20, [L"10^{-3.5}", L"10^{-4.5}", L"10^{-5.5}"]),
        yticks = ([-1, 0, 1], [L"\sum_%$i{xy}" for i in 1:3]),
        yticklabelrotation = pi/8,
        title = L"\int_0^1{x^2}",
        xlabel = L"\sum_k{x_k ⋅ y_k}",
        ylabel = L"\int_a^b{\sqrt{abx}}"
    ),
)

# AlgebraOfGraphics
df1 = DataFrame(a=0:0.01:10, c="red")
transform!(df1, :a => ByRow(x -> sin(3x) / (cos(x) + 2)) => :b)
df2 = DataFrame(a = 0:0.01:10,c="blue")
transform!(df2, :a => ByRow(x -> sin(x^2) / (cos(sqrt(x)) + 2)) => :b)
df = vcat(df1, df2)
plt = data(df) * mapping(:a, :b, color=:c => renamer("red" => L"\frac{\sin(3x)}{\cos(x) + 2}", "blue" => L"\frac{\sin(x^2)}{\cos(\sqrt{x}) + 2}")) * visual(Lines)
draw(plt)
