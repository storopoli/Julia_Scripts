# # Hello!
using Distributions, Plots, StatsPlots

# ## How to use `Literate.jl`?
# Run `Literate.markdown("Literate/plot_multivariate.jl", execute = true)`
# or Run `Literate.notebook("Literate/plot_multivariate.jl", execute = true)`

# ## Univariate Distributions using `StatsPlots`
plot(Normal())

# ## Multivariate Distributions using `Plots`
mvnorm = fit(MvNormal, [rand(0.0:100.0, 100) rand(0.0:100.0, 100)]')
Z = [pdf(mvnorm, [i, j]) for i = 0:100, j = 0:100]
plot(0:100, 0:100, Z, st = :wireframe, color = :blues)
plot(0:100, 0:100, Z, st = :surface)

# ## Environment
using InteractiveUtils
versioninfo()
using Pkg
Pkg.status()
