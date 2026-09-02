# UC_output_gap.py
# Gibbs sampler for a local linear trend (unobserved components) model that
# decomposes 100*log US real GDP into trend (potential) output and a cyclical
# output gap. The model is
#   y_t       = tau_t + c_t,
#   c_t       = phi_1*c_{t-1} + phi_2*c_{t-2} + u_t^c,   u_t^c   ~ N(0, sigc2),
#   D.tau_t   = D.tau_{t-1} + u_t^tau,                   u_t^tau ~ N(0, sigtau2),
# where D denotes the first difference, so trend growth follows a random walk.
# The trend tau is drawn with a precision (band-matrix) sampler; phi by an
# accept-reject step enforcing stationarity; sigc2 and (tau0, tau_{-1})
# from conjugate updates; and sigtau2 with a Griddy-Gibbs step via griddy_gibbs.py.
#
# Requires: griddy_gibbs.py, shade_nber_recessions.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.linalg import (cholesky, cholesky_banded, solveh_banded,
                          solve_banded, solve_triangular)

from griddy_gibbs import griddy_gibbs
from shade_nber_recessions import shade_nber_recessions

np.random.seed(42)
nsim = 20000
burnin = 1000
# load data - US GDP 1947Q1 - 2019Q4 (column GDPC1, first 292 rows)
data_raw = pd.read_csv('USGDP.csv')['GDPC1'].to_numpy()[:292]
data = 100*np.log(data_raw)
y = data
T = y.size

# prior hyperparameters
a0 = np.array([750., 750.])
B0 = 100*np.eye(2)
phi0 = np.array([1.3, -.7])
iVphi = np.eye(2)
nu_sigc2 = 3
S_sigc2 = 1*(nu_sigc2-1)
sigtau2_ub = .01

# storage
store_theta = np.zeros((nsim, 6))   # [phi, sigc2, sigtau2, tau0]
store_tau = np.zeros((nsim, T))
store_mu = np.zeros((nsim, T))      # annualized trend growth

# initialize
phi = np.array([1.34, -.7])
tau0 = np.array([y[0], y[0]])       # [tau_{0}, tau_{-1}]
sigc2 = .5
sigtau2 = .001

# construct a few things
S1 = sparse.csc_array((np.ones(T-1), (np.arange(1, T), np.arange(T-1))),
                      shape=(T, T))   # first-lag shift matrix
S2 = sparse.csc_array((np.ones(T-2), (np.arange(2, T), np.arange(T-2))),
                      shape=(T, T))   # second-lag shift matrix
H2 = sparse.eye_array(T, format='csc') - 2*S1 + S2
H2H2 = H2.T @ H2
Hphi = sparse.eye_array(T, format='csc') - phi[0]*S1 - phi[1]*S2
Xtau0 = np.column_stack((np.arange(2., T+2), -np.arange(1., T+1)))
n_grid = 500
count_phi = 0

# H2 is lower triangular with bandwidth 2: store it in banded form
# so that H2\b can be computed with a banded solve
H2_lb = np.zeros((3, T))
H2_lb[0, :] = 1.
H2_lb[1, :T-1] = -2.
H2_lb[2, :T-2] = 1.

for isim in range(nsim + burnin):
    # sample tau: Ktau has bandwidth 2, so use banded storage
    r = np.zeros(T)
    r[0] = 2*tau0[0] - tau0[1]
    r[1] = -tau0[0]
    alp_tau = solve_banded((2, 0), H2_lb, r)   # H2\r
    HpHp = Hphi.T @ Hphi
    Ktau = H2H2/sigtau2 + HpHp/sigc2
    ab = np.zeros((3, T))              # upper banded storage of Ktau
    ab[2, :] = Ktau.diagonal()
    ab[1, 1:] = Ktau.diagonal(1)
    ab[0, 2:] = Ktau.diagonal(2)
    tau_hat = solveh_banded(ab, H2H2 @ alp_tau/sigtau2 + HpHp @ y/sigc2)
    tau = tau_hat + solve_banded((0, 2), cholesky_banded(ab),
                                 np.random.randn(T))

    # sample phi
    c = y - tau
    Xphi = np.column_stack((np.concatenate(([0.], c[:T-1])),
                            np.concatenate(([0., 0.], c[:T-2]))))
    Kphi = iVphi + Xphi.T @ Xphi/sigc2
    phi_hat = np.linalg.solve(Kphi, iVphi @ phi0 + Xphi.T @ c/sigc2)
    phic = phi_hat + solve_triangular(cholesky(Kphi), np.random.randn(2))
    if phic.sum() < .99 and phic[1] - phic[0] < .99 and phic[1] > -.99:
        phi = phic
        Hphi = sparse.eye_array(T, format='csc') - phi[0]*S1 - phi[1]*S2
        count_phi = count_phi + 1

    # sample sigc2
    sigc2 = 1/np.random.gamma(nu_sigc2 + T/2,
                              1/(S_sigc2
                                 + (c - Xphi@phi) @ (c - Xphi@phi)/2))

    # sample sigtau2 via Griddy-Gibbs on (0, sigtau2_ub)
    del_tau = (np.concatenate(([tau0[0]], tau))
               - np.concatenate(([tau0[1], tau0[0]], tau[:-1])))
    # first differences of tau
    ddel_tau = del_tau[1:] - del_tau[:-1]
    # second differences of tau
    logf_sigtau2 = lambda x: (-(T/2)*np.log(x)
                              - (ddel_tau @ ddel_tau)/(2*x))
    sigtau2, _, _ = griddy_gibbs(logf_sigtau2, 1e-12,
                                 sigtau2_ub, n_grid)

    # sample tau0
    Ktau0 = np.linalg.solve(B0, np.eye(2)) \
        + Xtau0.T @ (H2H2 @ Xtau0)/sigtau2
    tau0_hat = np.linalg.solve(Ktau0, np.linalg.solve(B0, a0)
                               + Xtau0.T @ (H2H2 @ tau)/sigtau2)
    tau0 = tau0_hat + solve_triangular(cholesky(Ktau0), np.random.randn(2))

    if isim >= burnin:
        i = isim - burnin
        store_tau[i, :] = tau
        store_theta[i, :] = np.concatenate((phi, [sigc2, sigtau2], tau0))
        store_mu[i, :] = 4*(tau - np.concatenate(([tau0[0]], tau[:-1])))

tau_mean = store_tau.mean(axis=0)
theta_mean = store_theta.mean(axis=0)
theta_CI = np.quantile(store_theta, [.025, .975], axis=0)
mu_mean = store_mu.mean(axis=0)

print('posterior means of [phi1, phi2, sigc2, sigtau2, tau0, tau_{-1}]:')
print(theta_mean)
print('95% credible intervals (rows: 2.5%, 97.5%):')
print(theta_CI)

# plot of graphs
tt = np.arange(1947, 2020, .25)
fig1 = plt.figure(figsize=(9, 3.5))
y_gap = y - tau_mean
yl = [y_gap.min() - 1, y_gap.max() + 1]
shade_nber_recessions(yl[0], yl[1])

plt.plot(tt, y_gap, 'k', linewidth=1.5)
plt.plot(tt, np.zeros(T), '--k', linewidth=1)

plt.xlim(1947, 2020)
plt.ylim(yl)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Output gap', fontsize=14)
plt.tight_layout()

fig2 = plt.figure(figsize=(9, 3.5))
yl = [1, 4.5]
shade_nber_recessions(yl[0], yl[1])
plt.plot(tt, mu_mean, 'k', linewidth=1.5)
plt.xlim(1947, 2020)
plt.ylim(yl)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Output trend growth', fontsize=14)
plt.tight_layout()
fig1.savefig('UC_gap.eps')
fig2.savefig('UC_trend_growth.eps')
plt.show()
