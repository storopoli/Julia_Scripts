### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 94ae1a84-d3fd-11eb-0f06-917f894c2973
begin
	import Pkg
	Pkg.activate(mktempdir())
	Pkg.add([
			"DataFrames",
			"StatsPlots",
			"DifferentialEquations",
			"Plots",
			"LaTeXStrings",
			"PlutoUI",
			"Distributions"
			])
	using DataFrames
	using StatsPlots
	using Plots
	using LaTeXStrings
	using DifferentialEquations
	using PlutoUI
	using Distributions
	using Random
end

# ╔═╡ a9949aee-49bc-45bc-af4a-57197b1fcca6
md"""
# Data Generating Process for Twitter COVID-19 Symptom Epidemiological model
"""

# ╔═╡ 8deb582c-9ce5-4038-8163-7058d72a7d8f
md"""
The dynamics of SIRTD are governed by a system of ordinary differential equations (ODE):

$$\begin{align}
    \frac{dS}{dt} &= -\beta  S \frac{I}{N} \\
    \frac{dI}{dt} &= \beta  S  \frac{I}{N} - \frac{1}{d_I}  I \\
    \frac{dR}{dt} &= \frac{1}{d_I} I \left( 1 - \omega \right) \\
    \frac{dT}{dt} &= \frac{1}{d_I} I \omega - \frac{1}{d_T} T \\
    \frac{dD}{dt} &= \frac{1}{d_T} T 
\end{align}$$
"""

# ╔═╡ 4400f7a2-8819-4dee-aeb5-1e5634911ebb
md"""
where:

-  $S(t)$ is the number of people susceptible to becoming infected (no immunity);
-  $I(t)$ is the number of people currently infected (and infectious);
-  $T(t)$ is the number of terminally ill individuals who have been infected and will die;
-  $R(t)$ is the number of removed people (either dead or we assume they remain immune indefinitely);
-  $D(t)$ is the number of recovered people that unfortunately died;
-  $N = S(t) + I(t) + R(t) + T(t) + D(t)$ is the constant total number of individuals in the population;
-  $\beta$ is the constant rate of contacts between individuals per unit time that are sufficient to lead to transmission if one of the individuals is infectious and the other is susceptible;
-  $\omega$ is constant death rate of recovered individuals;
-  $d_I$ is the mean time for which individuals are infectious; and
-  $d_T$  is the mean time for which individuals are terminally-ill.

"""

# ╔═╡ 3248b45f-7dd2-489a-b4b6-187fe6bedb6d
"""
Data Generating Process for a SIRTD model in which the infected individuals have a probability of tweeting about their simptoms

## Arguments:
- `T::Int64`: number of days to simulate.
- `N::Int64`: total population.
- `λₜ::Float64`: daily probability that an infected individual tweet about his/hers symptoms.
- `C::Float64`: daily contacts that each infected individual has with other susceptible.
- `dᵢ::Int64`: number of days that an infected individual stays infected.
- `dₜ::Int64`: number of days that a terminally-ill individual stays terminally-ill before deceasing.
- `β::Float64`: infection rate
- `ω::Float64`: fatality rate
- `i₀::Int64`: number of infected individuals at the beggining of simulation
- `seed`: random seed for reproducibility

## Value
Returns a `Matrix{Int64}` of size `T`×6 where the columns are:
1. Susceptible
2. Infected
3. Recovered
4. Terminally-Ill
5. Deceased
6. Tweets
"""
function SIRTD_sim(T::Int64, N::Int64, λₜ::Float64, C::Float64, dᵢ::Int64, dₜ::Int64, β::Float64, ω::Float64, i₀::Int64, seed::Int64)
	Random.seed!(seed)
	s = N - i₀
	i = i₀
	r = 0
	t = 0
	d = 0
	tweets = 0
	state = zeros(Int64, T, 6)
	state[1, :] = [s i r t d tweets]
	for time ∈ 2:T
		# Tweets from Infected
		tweets = 0
		tweets = rand(Binomial(i, λₜ))
		# Susceptible <-> Infected
		daily_i = round(β*C*i / N*s)
		s = s -= daily_i
		i += daily_i
		# Infected <-> Recovered <-> Terminally-Ill
		daily_r = rand(Binomial(i, 1/dᵢ))
		daily_t = rand(Binomial(daily_r, ω))
		i -= daily_r
		r += (daily_r - daily_t)
		t += daily_t
		# Terminally-Ill <-> Deceased
		daily_d = rand(Binomial(t, 1/dₜ))
		t -= daily_d
		d += daily_d
		# Insert all into Matrix
		state[time, :] = [s i r t d tweets]
	end
	return state
end

# ╔═╡ d55cea2c-cb1f-4e17-aade-45da90e6951f
function SIRTD_ode!(du,u,p,t)
    (S,I,R,T,D) = u
    (C, dᵢ, dₜ, β, ω) = p
    N = S+I+R+T+D
    @inbounds begin
        du[1] = -β*C*I / N*S          # dS
        du[2] = β*C*I / N*S - 1/dᵢ*I  # dI
		du[3] = 1/dᵢ * I * (1-ω)      # dR
		du[4] = 1/dᵢ * I * ω - 1/dₜ*T # dT
        du[5] = 1/dₜ*T
    end
    nothing
end;

# ╔═╡ 4db55f8d-f84a-41db-ac18-17bfe93aafdb
md"""
N $(@bind N Slider(100:100:10_000, default=500, show_value=true))
T $(@bind T Slider(100:100:1_000, default=100, show_value=true))

β $(@bind β Slider(0.1:0.1:0.8, default=0.3, show_value=true))
ω $(@bind ω Slider(0.05:0.05:0.4, default=0.1, show_value=true))

λₜ $(@bind λₜ Slider(0.1:0.1:1.0, default=0.5, show_value=true))
dᵢ $(@bind dᵢ Slider(1:2:20, default=4, show_value=true))
dₜ $(@bind dₜ Slider(1:2:20, default=10, show_value=true))

i₀ $(@bind i₀ Slider(1:5:100, default=10, show_value=true))
C $(@bind C Slider(1.0:2.0:20.0, default=2.0, show_value=true))
"""

# ╔═╡ a036b478-2c7d-413d-af96-87fbb9bba70a
begin
	u0 = [N - i₀, i₀, 0, 0, 0] # S,I,R,T,D
	p = [C, dᵢ, dₜ, β, ω]
	tspan = (0, T)
	sim_data = SIRTD_sim(T, N, λₜ, float(C), dᵢ, dₜ, β, ω, i₀, 123)
	prob_ode = ODEProblem(SIRTD_ode!,u0,tspan,p)
	sol_ode = solve(prob_ode)
end

# ╔═╡ 9457171c-5393-42f4-8238-0e611090dc0f
begin
	plot(sol_ode, dpi=300, label=[L"S" L"I" L"R" L"T" L"D"], lw=3, la=0.7)
	plot!(sim_data, label=[L"S_{sim}" L"I_{sim}" L"R_{sim}" L"T_{sim}" L"D_{sim}" "tweets"], lw=3, ls=:dot, la=0.7)
	xlabel!(L"t")
	ylabel!("N")
end

# ╔═╡ Cell order:
# ╟─a9949aee-49bc-45bc-af4a-57197b1fcca6
# ╠═94ae1a84-d3fd-11eb-0f06-917f894c2973
# ╟─8deb582c-9ce5-4038-8163-7058d72a7d8f
# ╟─4400f7a2-8819-4dee-aeb5-1e5634911ebb
# ╠═3248b45f-7dd2-489a-b4b6-187fe6bedb6d
# ╠═d55cea2c-cb1f-4e17-aade-45da90e6951f
# ╠═a036b478-2c7d-413d-af96-87fbb9bba70a
# ╟─4db55f8d-f84a-41db-ac18-17bfe93aafdb
# ╠═9457171c-5393-42f4-8238-0e611090dc0f
