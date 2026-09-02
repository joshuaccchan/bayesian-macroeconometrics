# finmix_logchi2.py
# Gibbs sampler that fits a four-component normal mixture
#   y_t ~ sum_{m=1}^M w_m * N(mu_m, sig2_m)
# to simulated data from the log chi^2_1 distribution. Component
# indicators are sampled via the inverse-transform method, mixture
# weights from their Dirichlet full conditional (dirirnd.py), and
# (mu_m, sig2_m) from their normal-inverse-gamma full conditional.
# The chain is initialized using K-means clustering.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.cluster.vq import kmeans2

from dirirnd import dirirnd
from shaded_band import shaded_band

np.random.seed(42)

nsim = 20000
burnin = 1000
M = 4   # # of components

# generate data
T = 2000
df = 1
y = np.log(np.random.chisquare(df, T))


# true log-chi^2_1 density
def ftrue(x):
    return 1/np.sqrt(2*np.pi) * np.exp(0.5*x - 0.5*np.exp(x))


# prior hyperparameters
nu0 = 2
S0 = 1
mu0 = 0
Vmu = 100
a0 = 2*np.ones(M)

# grid
ngrid = 500
xgrid = np.linspace(-10, 5, ngrid)

store_mixden = np.zeros((nsim, ngrid))

# initialize using k-means (labels s are 0-based: 0,...,M-1)
mu, s = kmeans2(y[:, None], M, minit='++')
mu = mu.flatten()
sig2 = np.zeros(M)
alp = np.zeros(M)

for m in range(M):
    idx = (s == m)
    sig2[m] = np.var(y[idx], ddof=1)
    alp[m] = np.sum(idx)/T

for isim in range(nsim + burnin):
    # sample (mu_m, sig2_m) for each component
    for m in range(M):
        idx = (s == m)
        Tm = np.sum(idx)
        ym = y[idx]

        Kmum = 1/Vmu + Tm
        mum_hat = (mu0/Vmu + np.sum(ym))/Kmum
        Sm_hat = S0 + 0.5*(ym @ ym + mu0**2/Vmu
                           - mum_hat**2*Kmum)

        sig2[m] = 1/np.random.gamma(nu0 + Tm/2, 1/Sm_hat)
        mu[m] = mum_hat + np.sqrt(sig2[m]/Kmum)*np.random.randn()

    # sample s
    like = norm.pdf(y[:, None], mu[None, :], np.sqrt(sig2)[None, :])
    joint_den = like * alp[None, :]
    prob = joint_den / joint_den.sum(axis=1, keepdims=True)
    u = np.random.rand(T)
    cumprob = np.cumsum(prob, axis=1)
    s = np.sum(u[:, None] > cumprob, axis=1)

    # sample alp
    ns = np.zeros(M)
    for m in range(M):
        ns[m] = np.sum(s == m)
    alp = dirirnd(ns + a0, 1)[0]

    # store mixture density
    if isim >= burnin:
        isave = isim - burnin
        mixden = norm.pdf(xgrid[:, None], mu[None, :],
                          np.sqrt(sig2)[None, :]) @ alp
        store_mixden[isave, :] = mixden

mixden_mean = store_mixden.mean(axis=0)
mixden_low = np.percentile(store_mixden, 2.5, axis=0)
mixden_high = np.percentile(store_mixden, 97.5, axis=0)

print('Last draw of (mu_m, sig2_m, w_m) (component labels not identified):')
print(np.column_stack((mu, sig2, alp)))
print('Max abs deviation of posterior mean mixture density from truth:',
      np.max(np.abs(mixden_mean - ftrue(xgrid))))

plt.figure()

# 95% pointwise posterior credible band
shaded_band(xgrid, mixden_low, mixden_high, 0.85)

# posterior mean (solid black)
plt.plot(xgrid, mixden_mean, 'k-', linewidth=2, label='finite mixture')

# true density (dashed black)
plt.plot(xgrid, ftrue(xgrid), 'k--', linewidth=2, label='true density')

plt.xlim(-10, 5)
plt.xlabel(r'$y$', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend(loc='best', fontsize=14)
plt.show()
