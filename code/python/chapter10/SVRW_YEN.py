# SVRW_YEN.py
# Collapsed Gibbs sampler for the standard (random-walk) stochastic volatility
# model, fitted to daily YEN/USD returns. The model is
#   y_t = exp(h_t/2)*eps_t,   eps_t ~ N(0, 1),
#   h_t = h_{t-1} + u_t,      u_t   ~ N(0, sigh2),
# with priors h0 ~ N(a0, b0) and sigh2 ~ IG(nu_h, S_h).
#
# Requires: SVRW.py, SV_RW_gaussian_approx.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

from SVRW import SVRW
from SV_RW_gaussian_approx import SV_RW_gaussian_approx

np.random.seed(42)   # for reproducibility
nsim = 20000
burnin = 1000
data = pd.read_csv('YENUSD.csv', header=None).to_numpy().ravel()
y = data
T = y.size

# prior hyperparameters
a0 = 0
b0 = 100
nu_h = 3
S_h = .2**2*(nu_h-1)

# storage
store_theta = np.zeros((nsim, 2))   # [h0 sigh2]
store_h = np.zeros((nsim, T))

# precompute a few things
S1 = sparse.csc_array((np.ones(T-1), (np.arange(1, T), np.arange(T-1))),
                      shape=(T, T))
H = sparse.eye_array(T, format='csc') - S1
HH = H.T @ H
c = 1e-4
ystar = np.log(y**2 + c)

# initialize
sigh2 = .05
h0 = np.log(np.var(y, ddof=1))
h = SV_RW_gaussian_approx(y**2, h0, sigh2)
for isim in range(nsim + burnin):
    # sample sigh2
    sigh2 = 1/np.random.gamma(nu_h + T/2,
                              1/(S_h + (h-h0) @ (HH @ (h-h0))/2))

    # sample h0
    Kh0 = 1/b0 + 1/sigh2
    h0_hat = (a0/b0 + h[0]/sigh2)/Kh0
    h0 = h0_hat + 1/np.sqrt(Kh0)*np.random.randn()

    # sample h
    h = SVRW(ystar, h, h0, sigh2)

    if isim >= burnin:
        isave = isim - burnin
        store_h[isave, :] = h
        store_theta[isave, :] = [h0, sigh2]

h_std = np.exp(store_h/2)   # transform to standard deviation
h_mean = h_std.mean(axis=0)
h_CI = np.quantile(h_std, [0.05, 0.95], axis=0)
h_lower = h_CI[0, :]
h_upper = h_CI[1, :]

theta_mean = store_theta.mean(axis=0)
theta_CI = np.quantile(store_theta, [.05, .95], axis=0)

print('posterior means of [h0, sigh2]:', theta_mean)
print('90% credible intervals (rows: 5%, 95%):')
print(theta_CI)

tt = np.linspace(2005, 2013, T)
fig1 = plt.figure(figsize=(9, 3.5))
plt.plot(tt, y, 'k', linewidth=1.5)
plt.xlim(2005, 2013)
plt.xlabel('Time', fontsize=14)
plt.ylim(-6, 6)
plt.yticks(np.arange(-6, 7, 2))
plt.tight_layout()

fig2 = plt.figure(figsize=(9, 3.5))
plt.fill_between(tt, h_lower, h_upper, color=(0.8, 0.8, 0.8),
                 edgecolor='none')
plt.plot(tt, h_mean, 'k', linewidth=1.5)
plt.xlim(2005, 2013)
plt.xlabel('Time', fontsize=14)
plt.tight_layout()
fig1.savefig('YENdata.eps')
fig2.savefig('SV_YEN_h.eps')
plt.show()
