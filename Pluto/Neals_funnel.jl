### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ c52d6fbc-80eb-11eb-2e19-c98b6df84890
begin
	using PlutoUI, Distributions, Plots
	using PlutoUI: Print
	Plots.plotly()
end

# ╔═╡ 03839a78-80ee-11eb-0773-d3d97d413a99
begin
	using InteractiveUtils
	with_terminal() do
		versioninfo()
	end
end

# ╔═╡ 0c17e04c-80ec-11eb-1286-0b32f1ae4cf4
PlutoUI.TableOfContents(aside=true)

# ╔═╡ 0e8d6054-80ec-11eb-1c97-3dfc7626787d
html"<button onclick='present()'>present</button>"

# ╔═╡ 1840c79e-80ec-11eb-1088-89ed3d8b937b
md"# Neal's Funnel"

# ╔═╡ 5d402664-80ec-11eb-28bf-6bca9884388f
md"""
In this notebook we analyze Neal's Funnel (Neal, 2011) which defines a distribution that exemplifies the difficulties of sampling from some hierarchical models. Neal’s example has support for  $y \in \mathbb{R}$ and $x \in \mathbb{R}^2$ with density

```math
p(y,x) = \text{Normal}(y \mid 0,3) \times
\prod_{n=1}^2
\text{normal}\left(x_n \mid 0,\exp\left(\frac{y}{2}\right)\right).
```
"""

# ╔═╡ 697e0fd0-80ed-11eb-212d-8f0c4cb6d70c
md"## The Funnel"

# ╔═╡ db309090-80ec-11eb-043a-a94473ad9c0c
begin
	x = -2:0.01:2;
	kernel(x, y) = logpdf(Normal(0, exp(y / 2)), x)
	surface(x, x, kernel)
end

# ╔═╡ 75181872-80ed-11eb-0a7c-6596c1329943
md"## Reparameterization Trick"

# ╔═╡ 8009e3b4-80ed-11eb-13f5-3fe1d6ba097d
md"""
What if we reparameterize so that we can express $y$ and $x_n$ as standard normal distributions, by using a reparameterization trick[^2]:

```math
\begin{aligned}
x^* &\sim \text{Normal}(0, 0)\\
x &= x^* \cdot \sigma_x + \mu_x
\end{aligned}
```

[^2]: this also works for multivariate distributions.
"""

# ╔═╡ 9fc87404-80ed-11eb-06f0-af172ffbd824
md"""
## Applied to our example

We can provide the MCMC sampler a better-behaved posterior geometry to explore:

```math
\begin{aligned}
p(y^*,x^*) &= \text{Normal}(y^* \mid 0,0) \times \prod_{n=1}^9
\text{Normal}(x^*_n \mid 0,0)\\
y &= 3y^*\\
x_n &= \exp \left( \frac{y}{2} \right) x^*_n.
\end{aligned}
```
"""

# ╔═╡ c4f912d8-80ed-11eb-00c7-2732cfa17013
md"""
## Funnel Reparameterized

Below there is is the Neal's Funnel reparameterized as standard normal density in 3-D.
"""

# ╔═╡ ec7c9cee-80ed-11eb-138c-ffa24a4d9aa4
begin
	kernel_reparameterized(x, y) = logpdf(Normal(), x)
	surface(x, x,  kernel_reparameterized)
end

# ╔═╡ f98bbe94-80ed-11eb-1b15-6b0706f65397
md"""
## Environment
"""

# ╔═╡ Cell order:
# ╟─c52d6fbc-80eb-11eb-2e19-c98b6df84890
# ╟─0c17e04c-80ec-11eb-1286-0b32f1ae4cf4
# ╟─0e8d6054-80ec-11eb-1c97-3dfc7626787d
# ╟─1840c79e-80ec-11eb-1088-89ed3d8b937b
# ╟─5d402664-80ec-11eb-28bf-6bca9884388f
# ╟─697e0fd0-80ed-11eb-212d-8f0c4cb6d70c
# ╠═db309090-80ec-11eb-043a-a94473ad9c0c
# ╟─75181872-80ed-11eb-0a7c-6596c1329943
# ╟─8009e3b4-80ed-11eb-13f5-3fe1d6ba097d
# ╟─9fc87404-80ed-11eb-06f0-af172ffbd824
# ╟─c4f912d8-80ed-11eb-00c7-2732cfa17013
# ╠═ec7c9cee-80ed-11eb-138c-ffa24a4d9aa4
# ╟─f98bbe94-80ed-11eb-1b15-6b0706f65397
# ╟─03839a78-80ee-11eb-0773-d3d97d413a99
