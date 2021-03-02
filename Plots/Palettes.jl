using Plots

plot(Plots.fakedata(50, 5), w=3, palette=:Set1_5)
plot(Plots.fakedata(50, 5), w=3, palette=distinguishable_colors(5))
