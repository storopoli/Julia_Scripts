using Distributions, Plots, StatsPlots

# Univariate Normal - Using StatsPlots
norm = fit(Normal, rand(100))
plot(norm)
plot(Normal())

# Multivariate Normal - Using Plots
mvnorm = fit(MvNormal, [rand(0.0:100.0, 100) rand(0.0:100.0, 100)]')
Z = [pdf(mvnorm, [i, j]) for i in 0:100, j in 0:100];
plot(0:100, 0:100, Z, st=:wireframe, color=:blues)
plot(0:100, 0:100, Z, st=:surface)

# 3D Plots?
mvnorm = fit(MvNormal, [rand(0.0:100.0, 100) rand(0.0:100.0, 100)]')
Z = [pdf(mvnorm, [i, j]) for i in 0:100, j in 0:100]
plotly()
plot(0:100, 0:100, Z, st=:wireframe, color=:blues)
plot(0:100, 0:100, Z, st=:surface)
