from stannumpyro.dppl import NumPyroModel
from jax import random
import numpy as np

stanfile = "h_poisson.stan"
nd = 5
ns = 10
n = nd * ns
a0 = 1
a1 = 0.5
a0_sig = 0.3

y = np.zeros(n)
x = np.zeros(n)
idx = np.zeros(n)
i = 0

for s in range(ns):
    a0s = np.random.normal(0, 0.3)
    logpop = np.random.normal(9, 1.5)
    lambda_ = np.exp(a0 + a0s + a1 * logpop)
    for nd in range(nd):
        i += 1
        x[i] = logpop
        idx[i] = s
        y[i] = np.random.poisson(lambda_)

data = {
    'y': y,
    'x': x,
    'idx': idx,
    'N': n,
    'Ns': ns
}


model = NumPyroModel(stanfile)

 mcmc = model.mcmc(
        samples = 2000,
        warmups = 1000,
        chains=4
    )
