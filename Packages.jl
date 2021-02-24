using Pkg
Pkg.add([
    # Base
    "BenchmarkTools", "Pipe", "LaTeXStrings", "HTTP", "LazyArrays",
    # Plots
    "Plots", "StatsPlots", "PlotThemes",
    # Data
    "DataFrames",  "MLDataUtils", "Missings", "CategoricalArrays", "Tables", "CSV", "XLSX", "Arrow",
    # Stats/Bayesian
    "Distributions", "StatsFuns", "StatsBase", "Turing", "MCMCChains", "DynamicHMC", "AdvancedHMC", "DynamicPPL",
    # MultiThread
    "ThreadsX", "FLoops", "Transducers", "OnlineStats",
    # Graph
    "LightGraphs", "GraphIO", "GraphPlot", "GraphRecipes",
    # Presentation
    "Pluto", "PlutoUI", "Remark"
    ])

