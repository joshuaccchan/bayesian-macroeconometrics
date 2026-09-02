# UC_AR.py
# Gibbs sampler for a local level (unobserved components) model of US CPI
# inflation with an AR(1) cyclical component. The model is
#   y_t   = tau_t + eps_t,
#   eps_t = rho*eps_{t-1} + u_t,    u_t   ~ N(0, sig2),
#   tau_t = tau_{t-1} + eta_t,      eta_t ~ N(0, omega2),
# with priors tau0 ~ N(a0, b0), rho ~ U(-1, 1), and inverse-gamma priors on
# sig2 and omega2. The trend tau is drawn in one block from its Gaussian full
# conditional using a precision (band-matrix) sampler; rho is drawn from a
# truncated normal via tnormrnd.py; and sig2, omega2, and tau0 from standard
# conjugate updates.
#
# Requires: tnormrnd.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.linalg import cholesky_banded, solveh_banded, solve_banded

from tnormrnd import tnormrnd

np.random.seed(42)
nsim = 50000
burnin = 1000
# load data - US CPI 1948M1 - 2019M12 (column CPIAUCSL, first 864 rows)
data = pd.read_csv('USCPI.csv')['CPIAUCSL'].to_numpy()[:864]
y = data
T = y.size

# initialize for storage
store_tau = np.zeros((nsim, T))
store_theta = np.zeros((nsim, 4))   # [rho,sig2,omega2,tau0]

# prior hyperparameters
a0 = 5
b0 = 100
nu_sig0 = 3
S_sig0 = 1*(nu_sig0-1)
nu_omega0 = 3
S_omega0 = .25**2*(nu_omega0-1)

# initialize the Markov chain
sig2 = 1
omega2 = .1
tau0 = 5
rho = 0

# compute a few things outside the loop
S1 = sparse.csc_array((np.ones(T-1), (np.arange(1, T), np.arange(T-1))),
                      shape=(T, T))   # first-lag shift matrix
H = sparse.eye_array(T, format='csc') - S1
H_rho = sparse.eye_array(T, format='csc') - rho*S1
HH_rho = H_rho.T @ H_rho
HH = H.T @ H
HHiota = HH @ np.ones(T)

for isim in range(nsim + burnin):
    # sample tau: Ktau is tridiagonal, so use banded storage
    Ktau = HH/omega2 + HH_rho/sig2
    ab = np.zeros((2, T))            # upper banded storage of Ktau
    ab[1, :] = Ktau.diagonal()
    ab[0, 1:] = Ktau.diagonal(1)
    tau_hat = solveh_banded(ab, tau0/omega2*HHiota + HH_rho @ y/sig2)
    Ctau = cholesky_banded(ab)       # Ktau = Ctau'*Ctau with Ctau upper banded
    tau = tau_hat + solve_banded((0, 1), Ctau, np.random.randn(T))

    # sample rho
    e = y - tau
    Krho = np.sum(e[:T-1]**2)/sig2
    rho_hat = e[:T-1] @ e[1:]/np.sum(e[:T-1]**2)
    rho = tnormrnd(rho_hat, 1/Krho, -1, 1)
    H_rho = sparse.eye_array(T, format='csc') - rho*S1
    HH_rho = H_rho.T @ H_rho

    # sample sig2
    u = H_rho @ e
    sig2 = 1/np.random.gamma(nu_sig0 + T/2, 1/(S_sig0 + u @ u/2))

    # sample omega2
    omega2 = 1/np.random.gamma(nu_omega0 + T/2,
                               1/(S_omega0 + (tau-tau0) @ (HH @ (tau-tau0))/2))

    # sample tau0
    Ktau0 = 1/b0 + 1/omega2
    tau0_hat = (a0/b0 + tau[0]/omega2)/Ktau0
    tau0 = tau0_hat + np.sqrt(1/Ktau0)*np.random.randn()

    if isim >= burnin:
        isave = isim - burnin
        store_tau[isave, :] = tau
        store_theta[isave, :] = [rho, sig2, omega2, tau0]

tau_mean = store_tau.mean(axis=0)
theta_mean = store_theta.mean(axis=0)
tau_q = np.quantile(store_tau, [0.05, 0.95], axis=0)
tau_lo = tau_q[0, :]
tau_hi = tau_q[1, :]

print('posterior means of [rho, sig2, omega2, tau0]:')
print(theta_mean)

# monthly time axis: 1948M1 to 2019M12
tt = pd.date_range('1948-01-01', periods=T, freq='MS')

fig = plt.figure(figsize=(9, 3.5))
plt.fill_between(tt, tau_lo, tau_hi, color=(0.85, 0.85, 0.85),
                 edgecolor='none')
plt.plot(tt, tau_mean, 'k', linewidth=1.5, label='Trend')
plt.plot(tt, y, 'k:', linewidth=1, label='Inflation')

plt.xlim(tt[0] - pd.DateOffset(months=12), tt[-1] + pd.DateOffset(months=12))
plt.ylim(min(y.min(), tau_lo.min()) - 1, max(y.max(), tau_hi.max()) + 1)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Inflation', fontsize=14)
plt.legend(fontsize=14, loc='best')
plt.tight_layout()
fig.savefig('CPI_trend.eps')
plt.show()
