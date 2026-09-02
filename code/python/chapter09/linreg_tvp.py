# linreg_tvp.py
# Gibbs sampler for a time-varying parameter (TVP) regression, applied to a
# Phillips curve for US PCE inflation. The model is
#   y_t    = x_t'*beta_t + eps_t,     eps_t ~ N(0, sig2),
#   beta_t = beta_{t-1} + eta_t,      eta_t ~ N(0, Omega),
# with x_t = (1, gap_t, y_{t-1})' and Omega = diag(omega_1^2,...,omega_k^2), so
# each coefficient follows an independent random walk.
# The stacked coefficient path beta is drawn in one block from its Gaussian
# full conditional, whose block-tridiagonal precision matrix uses the
# Kronecker structure (H'H) kron Omega^{-1}; the remaining blocks
# (sig2, omega_j^2, beta0) use standard conjugate updates.
# The stacked design matrix Z = diag(x_1',...,x_T') is formed by SURform.py.
#
# Requires: SURform.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.linalg import (cholesky, cholesky_banded, solveh_banded,
                          solve_banded, solve_triangular)

from SURform import SURform

np.random.seed(42)
nsim = 20000
burnin = 1000

# load data (columns PCECTPI and Output Gap, first 240 rows: 1960Q1-2019Q4)
data = pd.read_csv('USPCE_OutputGap.csv')[['PCECTPI', 'Output Gap']] \
    .to_numpy()[:240]
infl = data[:, 0]   # PCE inflation
gap = data[:, 1]    # output gap

# construct y and X
y = infl[1:]
g = gap[1:]
ylag = infl[:-1]    # y_{t-1}
T = y.size
X = np.column_stack((np.ones(T), g, ylag))
k = X.shape[1]

# prior hyperparameters
beta00 = np.zeros(k)
iVbeta0 = 1/100*np.eye(k)
nu_sig = 3
S_sig = 1                            # IG prior for sigma^2
nu_om = 3                            # IG prior for omega_j^2
S_om = np.array([0.125, 0.025, 0.025])**2*(nu_om-1)

# initialize chain
beta0 = np.zeros(k)
beta_ols = np.linalg.solve(X.T @ X, X.T @ y)
sig2 = np.mean((y - X @ beta_ols)**2)
omega2 = 0.01**2*np.ones(k)

# precompute a few things
S1 = sparse.csc_array((np.ones(T-1), (np.arange(1, T), np.arange(T-1))),
                      shape=(T, T))
H = sparse.eye_array(T, format='csc') - S1
HH = H.T @ H
Z = SURform(X)
ZZ = Z.T @ Z
Zy = Z.T @ y

# storage
store_beta = np.zeros((nsim, T*k))
# [beta0', sig2, omega2']
store_theta = np.zeros((nsim, 2*k + 1))

for isim in range(nsim + burnin):
    # sample beta: Kbeta has bandwidth k, so use banded storage
    iOmega = sparse.diags_array(1/omega2)
    P = sparse.kron(HH, iOmega, format='csc')   # prior precision
    Kbeta = P + ZZ/sig2
    ab = np.zeros((k+1, k*T))          # upper banded storage of Kbeta
    for j in range(k+1):
        ab[k-j, j:] = Kbeta.diagonal(j)
    beta_hat = solveh_banded(ab, P @ np.tile(beta0, T) + Zy/sig2)
    beta = beta_hat + solve_banded((0, k), cholesky_banded(ab),
                                   np.random.randn(k*T))

    # sample sigma^2
    e = y - Z @ beta
    sig2 = 1/np.random.gamma(nu_sig + T/2, 1/(S_sig + e @ e/2))

    # sample omega_j^2
    Beta = beta.reshape(T, k)          # row t is beta_t'
    SSE = np.sum((Beta - np.vstack((beta0, Beta[:T-1, :])))**2, axis=0)
    omega2 = 1/np.random.gamma(nu_om + T/2, 1/(S_om + 0.5*SSE))

    # sample beta0
    Kbeta0 = iVbeta0 + np.diag(1/omega2)
    beta0_hat = np.linalg.solve(Kbeta0, iVbeta0 @ beta00
                                + beta[:k]/omega2)
    beta0 = beta0_hat + solve_triangular(cholesky(Kbeta0),
                                         np.random.randn(k))

    if isim >= burnin:
        isave = isim - burnin
        store_beta[isave, :] = beta
        store_theta[isave, :] = np.concatenate((beta0, [sig2], omega2))

Beta_mean = store_beta.mean(axis=0).reshape(T, k)
Beta_q = np.quantile(store_beta, [0.05, 0.95], axis=0) \
    .reshape(2, T, k).transpose(1, 2, 0)

theta_mean = store_theta.mean(axis=0)
print("posterior means of [beta0', sig2, omega2']:")
print(theta_mean)

# plot coefficients with 90% CI
tt = np.arange(1960.25, 2020, .25)

names = ['Intercept', 'Output gap', 'Lagged inflation']

fig = plt.figure(figsize=(6, 5))
for j in range(k):
    plt.subplot(k, 1, j+1)
    lo = Beta_q[:, j, 0]
    hi = Beta_q[:, j, 1]
    plt.fill_between(tt, lo, hi, color=(0.85, 0.85, 0.85), edgecolor='none')

    plt.plot(tt, Beta_mean[:, j], 'k', linewidth=1.5)

    plt.ylabel(names[j], fontsize=12)
    if j == k-1:
        plt.xlabel('Time')
plt.tight_layout()
fig.savefig('tvp_phillips_curve.eps')
plt.show()
