# https://github.com/TuringLang/TuringExamples/tree/master/benchmarks/h_poisson
using Memoization, LazyArrays, Turing, DynamicHMC
using Distributions: Normal, Poisson
using Random:seed!
seed!(1)
Threads.nthreads()

# Get Data
function get_data(nd=5, ns=10, a0=1, a1=0.5, a0_sig=0.3)
    n = nd * ns
    y = zeros(Int, n)
    x = zeros(n)
    idx = similar(y)
    i = 0
    for s in 1:ns
        a0s = rand(Normal(0, a0_sig))
        logpop = rand(Normal(9, 1.5))
        λ = exp(a0 + a0s + a1 * logpop)
        for nd in 1:nd
            i += 1
            x[i] = logpop
            idx[i] = s
            y[i] = rand(Poisson(λ))
        end
    end
    return Dict(
        "y" => y,
        "x" => x,
        "idx" => idx,
        "N" => n,
        "Ns" => ns,
    )
end

data = get_data()


# Turing
lazyarray(f, x) = LazyArray(Base.broadcasted(f, x))
@model h_poisson(y, x, idx, N, Ns) = begin
    a0 ~ Normal(0, 10)
    a1 ~ Normal(0, 1)
    a0_sig ~ truncated(Cauchy(0, 1), 0, Inf)
    a0s ~ filldist(Normal(0, a0_sig), Ns)
    alpha = a0 .+ a0s[idx] .+ a1 * x
    y ~ arraydist(lazyarray(LogPoisson, alpha))
end

model = h_poisson(data["y"], data["x"], data["idx"], data["N"], data["Ns"])

# 78s (sampling 50s) ESS a0_sig 1128
@time chn = sample(model, NUTS(), MCMCThreads(), 2_000, 4)

# 64s (sampling 57s) ESS a0_sig 1007
@time chn = sample(model, DynamicNUTS(), MCMCThreads(), 2_000, 4)
