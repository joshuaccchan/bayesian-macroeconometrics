# UC_SVM.py
# Metropolis-within-Gibbs sampler for the unobserved components stochastic
# volatility in mean model with time-varying coefficients, fitted to quarterly
# PCE inflation. The measurement equation is
#   y_t = tau_t + alp_t*exp(h_t) + eps_t,   eps_t ~ N(0, exp(h_t)),
# where the coefficient vector gam_t = (tau_t, alp_t)' follows a random walk and
# the log-volatility h_t is a random walk. The coefficient path gam and the
# static parameters use conjugate updates; h is sampled by the Laplace-based
# acceptance-rejection MH step (sample_SVM_h_ARMH.py).
# The stacked design matrix is built with SURform.py.
#
# Requires: sample_SVM_h_ARMH.py, SURform.py, SV_RW_gaussian_approx.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.linalg import (cholesky, cholesky_banded, solveh_banded,
                          solve_banded, solve_triangular)

from sample_SVM_h_ARMH import sample_SVM_h_ARMH
from SURform import SURform
from SV_RW_gaussian_approx import SV_RW_gaussian_approx

np.random.seed(42)   # for reproducibility
nsim = 20000
burnin = 5000

# load PCE data - 1960Q1-2024Q4 (column PCECTPI, first 260 rows)
data = pd.read_csv('USPCE.csv')['PCECTPI'].to_numpy()[:260]
y = data
T = y.size

# prior hyperparameters
agam = np.array([2., 0.])
iVgam = np.eye(2)/100
nuOmega = 3
SOmega = (nuOmega-1)*np.array([0.25**2, 0.10**2])
nuh = 3
Sh = 0.2**2*(nuh - 1)
ah = 0
Vh = 100

# initialize
omega = np.array([0.25**2, 0.10**2])   # store the diagonal elements
sigh2 = 0.2**2
h0 = np.log(np.var(y, ddof=1))
tau = np.mean(y)*np.ones(T)
h = SV_RW_gaussian_approx((y - tau)**2, h0, sigh2)
exp_h = np.exp(h)
gam0 = np.zeros(2)

# storage
store_theta = np.zeros((nsim, 6))   # [h0 sigh2 omega' gam0']
store_tau = np.zeros((nsim, T))
store_alp = np.zeros((nsim, T))
store_h = np.zeros((nsim, T))

# precompute fixed matrices
S1gam = sparse.csc_array((np.ones(2*(T-1)),
                          (np.arange(2, 2*T), np.arange(2*(T-1)))),
                         shape=(2*T, 2*T))
Hgam = sparse.eye_array(2*T, format='csc') - S1gam
S1 = sparse.csc_array((np.ones(T-1), (np.arange(1, T), np.arange(T-1))),
                      shape=(T, T))
H = sparse.eye_array(T, format='csc') - S1
HH = H.T @ H
accept_h = 0

for isim in range(nsim + burnin):
    # sample gam: Kgam has bandwidth 2, so use banded storage
    Xgam = SURform(np.column_stack((np.ones(T), exp_h)))
    tmp = Xgam.T @ sparse.diags_array(1/exp_h, format='csc')
    # prior precision
    Pgam = Hgam.T @ sparse.kron(sparse.eye_array(T, format='csc'),
                                sparse.diags_array(1/omega)) @ Hgam
    Kgam = Pgam + tmp @ Xgam
    ab = np.zeros((3, 2*T))          # upper banded storage of Kgam
    ab[2, :] = Kgam.diagonal()
    ab[1, 1:] = Kgam.diagonal(1)
    ab[0, 2:] = Kgam.diagonal(2)
    gamhat = solveh_banded(ab, Pgam @ np.tile(gam0, T) + tmp @ y)
    gam = gamhat + solve_banded((0, 2), cholesky_banded(ab),
                                np.random.randn(2*T))
    tau = gam[0::2]
    alp = gam[1::2]

    # sample gam0
    Kgam0 = iVgam + np.diag(1/omega)
    gam0hat = np.linalg.solve(Kgam0, iVgam @ agam + gam[:2]/omega)
    gam0 = gam0hat + solve_triangular(cholesky(Kgam0), np.random.randn(2))

    # sample h using acceptance-rejection MH
    h, accept = sample_SVM_h_ARMH(y, alp, tau, h, h0, sigh2, HH)
    exp_h = np.exp(h)
    if isim >= burnin:
        accept_h = accept_h + accept

    # sample Omega
    e_gam = (gam - np.concatenate((gam0, gam[:-2]))).reshape(T, 2)
    newSOmega = SOmega + np.sum(e_gam**2, axis=0)/2
    omega = 1/np.random.gamma(nuOmega + T/2, 1/newSOmega)

    # sample sigh2
    e_h = np.concatenate(([h[0]-h0], np.diff(h)))
    newSh = Sh + np.sum(e_h**2)/2
    sigh2 = 1/np.random.gamma(nuh + T/2, 1/newSh)

    # sample h0
    Kh0 = 1/Vh + 1/sigh2
    h0hat = (ah/Vh + h[0]/sigh2)/Kh0
    h0 = h0hat + np.random.randn()/np.sqrt(Kh0)

    # store draws
    if isim >= burnin:
        isave = isim - burnin
        store_tau[isave, :] = tau
        store_alp[isave, :] = alp
        store_h[isave, :] = h
        store_theta[isave, :] = np.concatenate(([h0, sigh2], omega, gam0))

print(f'Acceptance rate for h = {accept_h/nsim:.3f}')
# posterior summaries
tau_mean = store_tau.mean(axis=0)
tau_CI = np.quantile(store_tau, [0.05, 0.95], axis=0)
tau_lower = tau_CI[0, :]
tau_upper = tau_CI[1, :]

alp_mean = store_alp.mean(axis=0)
alp_CI = np.quantile(store_alp, [0.05, 0.95], axis=0)
alp_lower = alp_CI[0, :]
alp_upper = alp_CI[1, :]

theta_mean = store_theta.mean(axis=0)
print("posterior means of [h0, sigh2, omega_tau2, omega_alp2, tau00, alp00]:")
print(theta_mean)

# plot alpha_t with 90% CI
tt = np.linspace(1960, 2024.75, T)
fig = plt.figure(figsize=(9, 3))
plt.fill_between(tt, alp_lower, alp_upper, color=(0.8, 0.8, 0.8),
                 edgecolor='none')
plt.plot(tt, alp_mean, 'k', linewidth=1.5)
plt.xlim(tt.min() - 1, tt.max() + 1)
plt.tight_layout()
fig.savefig('UC_SVM.eps')
plt.show()
