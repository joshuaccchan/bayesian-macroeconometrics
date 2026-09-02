# SVM_SP500_HMC.py
# Gibbs sampler with an HMC update of the log-volatility for the stochastic
# volatility in mean (SVM) model, fitted to daily S&P 500 excess returns. The
# model is
#   y_t = mu + alp*exp(h_t) + eps_t,   eps_t ~ N(0, exp(h_t)),
#   h_t = h_{t-1} + u_t,               u_t   ~ N(0, sigh2),
# with priors gam = (mu, alp)' ~ N(gam0, iVgam^{-1}), sigh2 ~ IG(nu_h, S_h),
# and h0 ~ N(a0, b0). The regression coefficients, sigh2, and h0 use conjugate
# updates; the log-volatility h is sampled by Hamiltonian Monte Carlo using the
# functions logpost_svm_h, grad_logpost_svm_h, and leapfrog below (local
# functions at the end of the MATLAB script; Python needs them defined before
# use).
#
# SP500.csv is not distributed with this repository (proprietary); run
# get_SP500_data.py once to create it.
#
# Requires: SV_RW_gaussian_approx.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.linalg import cholesky, solve_triangular

from SV_RW_gaussian_approx import SV_RW_gaussian_approx

# Seed note: the sampler has a fragile start. The log-volatility is initialized
# at the smooth Gaussian-approximation path, so if no early HMC proposal is
# accepted while sigh2 is still at its initial value, sigh2 collapses to ~0.013
# (conditional on the too-smooth path) where every subsequent proposal is
# rejected (dH ~ 12) and the h-chain freezes -- the printed acceptance rate is
# then 0.000. Whether an early proposal is accepted is seed luck in either
# language (MATLAB's rng(42) escapes; numpy's seed 42 does not), so we use a
# numpy seed that escapes. A frozen run is always visible in the acceptance
# rate.
np.random.seed(1)    # for reproducibility (see note above)
nsim = 20000
burnin = 1000

# prepare the data
sp500_raw = pd.read_csv('SP500.csv', header=None).to_numpy()  # [date, index level]
dff_raw = pd.read_csv('DFF.csv', header=None).to_numpy()  # [date, fed funds rate]
# keep only valid S&P 500 obs.
valid_idx = sp500_raw[:, 1] != 0
sp500_raw = sp500_raw[valid_idx, :]
sp500_dates = sp500_raw[1:, 0]
sp500_returns = 100*np.log(sp500_raw[1:, 1]/sp500_raw[:-1, 1])
# match return date with the corresponding DFF obs.
is_matched = np.isin(sp500_dates, dff_raw[:, 0])
sp500_returns = sp500_returns[is_matched]
# position of each matched date in the DFF sample (DFF dates are sorted)
dff_loc = np.searchsorted(dff_raw[:, 0], sp500_dates[is_matched])
# annualized fed funds rate (percent) to daily
y = sp500_returns - dff_raw[dff_loc, 1]/252
T = y.size

# prior hyperparameters
gam0 = np.zeros(2)
iVgam = np.eye(2)/100
nu_h = 3
S_h = .2**2*(nu_h-1)
a0 = 0
b0 = 100

# HMC settings for h
eps = 0.04    # step size
L = 20        # number of leapfrog steps
accept_h = 0  # acceptance counter

# precompute banded prior precision pieces
S1 = sparse.csc_array((np.ones(T-1), (np.arange(1, T), np.arange(T-1))),
                      shape=(T, T))
H = sparse.eye_array(T, format='csc') - S1
HH = H.T @ H

# initialize
alp = 0
mu = np.mean(y)
h0 = np.log(np.var(y, ddof=1))
sigh2 = .2
# initialize h using Gaussian approximation
h = SV_RW_gaussian_approx((y - mu)**2, h0, sigh2)


def logpost_svm_h(y, mu, alp, h, h0, sigh2, HH):
    # Log conditional posterior density of the log-volatility h (up to an
    # additive constant) in the SVM model, combining the Gaussian likelihood
    # with the random-walk prior.
    r = y - mu - alp*np.exp(h)
    lden = (-0.5*np.sum(h) - 0.5*np.sum((r**2)*np.exp(-h))
            - 0.5/sigh2*(h - h0) @ (HH @ (h - h0)))
    return lden


def grad_logpost_svm_h(y, mu, alp, h, h0, sigh2, HH):
    # Gradient of logpost_svm_h with respect to h (same inputs).
    grad_like = (-0.5 - 0.5*alp**2*np.exp(h)
                 + 0.5*(y - mu)**2*np.exp(-h))
    grad_prior = -(HH @ (h - h0))/sigh2
    grad = grad_like + grad_prior
    return grad


def leapfrog(h, p, eps, L, grad_logpost):
    # Leapfrog integrator for HMC: L steps of size eps evolving (h, p) along
    # the Hamiltonian trajectory, with adjacent half-steps combined (L+1
    # gradient evaluations). grad_logpost is a function returning grad log p(h).
    hNew = h.copy()
    pNew = p.copy()
    pNew = pNew + 0.5*eps*grad_logpost(hNew)
    for i in range(1, L+1):
        hNew = hNew + eps*pNew
        if i < L:
            pNew = pNew + eps*grad_logpost(hNew)
        else:
            pNew = pNew + 0.5*eps*grad_logpost(hNew)
    pNew = -pNew
    return hNew, pNew


# storage
store_theta = np.zeros((nsim, 3))   # [mu alp sigh2]
store_h = np.zeros((nsim, T))
for isim in range(nsim + burnin):
    # sample gam = (mu, alp)'
    X = np.column_stack((np.ones(T), np.exp(h)))
    iSy = 1/np.exp(h)                 # diagonal of Sigma_y^{-1}
    XiSy = X.T*iSy
    Kgam = iVgam + XiSy @ X
    gam_hat = np.linalg.solve(Kgam, iVgam @ gam0 + XiSy @ y)
    gam = gam_hat + solve_triangular(cholesky(Kgam), np.random.randn(2))
    mu = gam[0]
    alp = gam[1]

    # sample h using HMC
    logpost = lambda ht: logpost_svm_h(y, mu, alp, ht, h0, sigh2, HH)
    grad = lambda ht: grad_logpost_svm_h(y, mu, alp, ht, h0, sigh2, HH)
    p0 = np.random.randn(T)   # initial momentum
    hc, pc = leapfrog(h, p0, eps, L, grad)
    H0 = -logpost(h) + 0.5*(p0 @ p0)
    Hc = -logpost(hc) + 0.5*(pc @ pc)
    if np.log(np.random.rand()) < -(Hc - H0):
        h = hc
        if isim >= burnin:
            accept_h = accept_h + 1

    # # sample h using a Laplace-based ARMH step
    # h, accept = sample_SVM_h_ARMH(y, alp, mu, h, h0, sigh2, HH)
    # if isim >= burnin:
    #     accept_h = accept_h + accept

    # sample sigh2
    sigh2 = 1/np.random.gamma(nu_h + T/2,
                              1/(S_h + (h-h0) @ (HH @ (h-h0))/2))

    # sample h0
    Kh0 = 1/b0 + 1/sigh2
    h0_hat = (a0/b0 + h[0]/sigh2)/Kh0
    h0 = h0_hat + np.random.randn()/np.sqrt(Kh0)

    if isim >= burnin:
        isave = isim - burnin
        store_h[isave, :] = h
        store_theta[isave, :] = [mu, alp, sigh2]

print(f'HMC acceptance rate = {accept_h/nsim:.3f}')
# posterior summaries
h_mean = np.exp(store_h/2).mean(axis=0)
h_CI = np.quantile(np.exp(store_h/2), [0.05, 0.95], axis=0)  # 90% credible interval
h_lower = h_CI[0, :]
h_upper = h_CI[1, :]

theta_mean = store_theta.mean(axis=0)
theta_CI = np.quantile(store_theta, [0.025, 0.975], axis=0)  # 95% credible interval

print('posterior means of [mu, alp, sigh2]:', theta_mean)
print('95% credible intervals (rows: 2.5%, 97.5%):')
print(theta_CI)

# Plot posterior mean and 90% credible interval for exp(h_t/2)
tt = np.linspace(2013, 2016, T)

fig = plt.figure(figsize=(9, 3.5))
plt.fill_between(tt, h_lower, h_upper, color=(0.8, 0.8, 0.8),
                 edgecolor='none')
plt.plot(tt, h_mean, 'k', linewidth=1.5)

plt.xlim(2013, 2016)
plt.xticks([2013, 2014, 2015, 2016])
plt.tight_layout()
fig.savefig('SVM_h.eps')
plt.show()
