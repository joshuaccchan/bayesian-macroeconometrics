# UCSV.py
# Gibbs sampler for the unobserved components model with stochastic volatility
# of Stock and Watson (2007), fitted to quarterly PCE inflation. The
# model is
#   y_t   = tau_t + eps^y_t,         eps^y_t   ~ N(0, exp(h_t)),
#   tau_t = tau_{t-1} + eps^tau_t,   eps^tau_t ~ N(0, exp(g_t)),
# with random-walk log-volatilities h_t (gap) and g_t (trend), each with an
# inverse-gamma prior on its innovation variance.
#
# Requires: SVRW.py, SV_RW_gaussian_approx.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.linalg import cholesky_banded, solveh_banded, solve_banded

from SVRW import SVRW
from SV_RW_gaussian_approx import SV_RW_gaussian_approx

np.random.seed(42)   # for reproducibility
nsim = 50000
burnin = 1000

# load PCE data - 1960Q1-2024Q4 (column PCECTPI, first 260 rows)
data = pd.read_csv('USPCE.csv')['PCECTPI'].to_numpy()[:260]
y = data
T = y.size

# prior hyperparameters
a0_h = 0
b0_h = 10    # h0 ~ N(a0_h, b0_h)
a0_g = 0
b0_g = 10    # g0 ~ N(a0_g, b0_g)
a0_tau = 0
b0_tau = 10  # tau0 ~ N(a0_tau, b0_tau)
nu_oh = 3
S_oh = 0.2**2*(nu_oh-1)   # omega_h^2 ~ IG(nu_oh, S_oh)
nu_og = 3
S_og = 0.2**2*(nu_og-1)   # omega_g^2 ~ IG(nu_og, S_og)

# precompute a few things
c = 1e-4   # log-squared safeguard
S1 = sparse.csc_array((np.ones(T-1), (np.arange(1, T), np.arange(T-1))),
                      shape=(T, T))
H = sparse.eye_array(T, format='csc') - S1
HH = H.T @ H

# initialize
tau0 = np.mean(y)
tau = tau0*np.ones(T)
h0 = np.log(np.var(y, ddof=1))
g0 = np.log(np.var(y, ddof=1))
omega_h2 = 0.1
omega_g2 = 0.1
# initialize h using Gaussian approximation
h = SV_RW_gaussian_approx((y - tau)**2, h0, omega_h2)
# initialize g using Gaussian approximation
dtau = tau - np.concatenate(([tau0], tau[:-1]))
g = SV_RW_gaussian_approx(dtau**2, g0, omega_g2)

# storage
# [omega_h2 omega_g2 h0 g0 tau0]
store_theta = np.zeros((nsim, 5))
store_tau = np.zeros((nsim, T))
store_h = np.zeros((nsim, T))
store_g = np.zeros((nsim, T))

for isim in range(nsim + burnin):

    # sample tau: Ktau is tridiagonal, so use banded storage
    iOh = np.exp(-h)                     # diagonal of Omega_h^{-1}
    HiOgH = H.T @ sparse.diags_array(np.exp(-g), format='csc') @ H
    Ktau = HiOgH + sparse.diags_array(iOh, format='csc')
    ab = np.zeros((2, T))                # upper banded storage of Ktau
    ab[1, :] = Ktau.diagonal()
    ab[0, 1:] = Ktau.diagonal(1)
    tau_mean = solveh_banded(ab, HiOgH @ (tau0*np.ones(T)) + iOh*y)
    tau = tau_mean + solve_banded((0, 1), cholesky_banded(ab),
                                  np.random.randn(T))

    # sample tau0
    Ktau0 = 1/b0_tau + np.exp(-g[0])
    tau0_hat = (a0_tau/b0_tau + tau[0]*np.exp(-g[0]))/Ktau0
    tau0 = tau0_hat + (1/np.sqrt(Ktau0))*np.random.randn()

    # sample h
    ystar_h = np.log((y - tau)**2 + c)
    h = SVRW(ystar_h, h, h0, omega_h2)

    # sample omega_h2
    omega_h2 = 1/np.random.gamma(nu_oh + T/2,
                                 1/(S_oh + 0.5*(h - h0) @ (HH @ (h - h0))))

    # sample h0
    Kh0 = 1/b0_h + 1/omega_h2
    h0_hat = (a0_h/b0_h + h[0]/omega_h2)/Kh0
    h0 = h0_hat + (1/np.sqrt(Kh0))*np.random.randn()

    # sample g
    dtau = tau - np.concatenate(([tau0], tau[:-1]))
    ystar_g = np.log(dtau**2 + c)
    g = SVRW(ystar_g, g, g0, omega_g2)

    # sample omega_g2
    omega_g2 = 1/np.random.gamma(nu_og + T/2,
                                 1/(S_og + 0.5*(g - g0) @ (HH @ (g - g0))))

    # sample g0
    Kg0 = 1/b0_g + 1/omega_g2
    g0_hat = (a0_g/b0_g + g[0]/omega_g2)/Kg0
    g0 = g0_hat + (1/np.sqrt(Kg0))*np.random.randn()

    if isim >= burnin:
        isave = isim - burnin
        store_tau[isave, :] = tau
        store_h[isave, :] = h
        store_g[isave, :] = g
        store_theta[isave, :] = [omega_h2, omega_g2, h0, g0, tau0]

# posterior summaries
theta_mean = store_theta.mean(axis=0)
tau_mean = store_tau.mean(axis=0)
h_mean = np.exp(store_h/2).mean(axis=0)
g_mean = np.exp(store_g/2).mean(axis=0)

print('posterior means of [omega_h2, omega_g2, h0, g0, tau0]:')
print(theta_mean)

tt = np.arange(1960, 2025, .25)
fig1 = plt.figure(figsize=(9, 3.5))
plt.plot(tt, y, 'k--', linewidth=1.2, label='Actual inflation')   # data
plt.plot(tt, tau_mean, 'k', linewidth=1.8, label='Trend inflation')  # trend

plt.xlim(1959.75, 2025)
plt.xlabel('Time', fontsize=14)
plt.legend(fontsize=12, loc='best')
plt.tight_layout()

fig2 = plt.figure(figsize=(9, 3.5))
plt.plot(tt, h_mean, 'k', linewidth=1.8, label='Gap volatility')
plt.plot(tt, g_mean, 'k--', linewidth=1.5, label='Trend volatility')

plt.xlim(1959.75, 2025)
plt.xlabel('Time', fontsize=14)
plt.legend(fontsize=12, loc='best')
plt.tight_layout()
fig1.savefig('UCSV_trend.eps')
fig2.savefig('UCSV_SV.eps')
plt.show()
