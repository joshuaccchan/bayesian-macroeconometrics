# DFM.py
# Gibbs sampler for the dynamic factor model fitted to a large monthly
# macroeconomic panel (FRED-MD). The model is
#   y_t = A f_t + eps_t,                              eps_t ~ N(0, Sigma),
#   f_t = Phi_1 f_{t-1} + ... + Phi_p f_{t-p} + u_t,  u_t   ~ N(0, Omega),
# with A lower triangular (ones on the diagonal) and Sigma, Omega diagonal.
# This script estimates the one-factor, one-lag specification (r=1, p=1) as a
# business-cycle indicator. The 4-block Gibbs sampler draws the factor path f,
# the free loadings a, the variances (Sigma, Omega), and the AR coefficients
# phi (drawn from a normal truncated to the stationary region). It plots the
# posterior mean factor with NBER recession shading.
#
# Requires: shade_nber_recessions.py

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import cholesky_banded, solve_banded, solveh_banded, \
    solve_triangular
import matplotlib.pyplot as plt

from shade_nber_recessions import shade_nber_recessions

np.random.seed(42)   # for reproducibility

nsim = 20000
burnin = 1000
r = 1   # number of factors

# load data
raw = pd.read_csv('FRED-MD.csv')
dates_num = raw.iloc[:, 0].to_numpy()   # first column contains dates as year + month/12
month_idx = np.round(12*dates_num).astype(int)   # months since year 0
year_part = (month_idx - 1)//12
month_part = month_idx - 12*year_part
dates = pd.to_datetime({'year': year_part, 'month': month_part, 'day': 1})
data = raw.iloc[:, 1:].to_numpy()   # remaining columns are data
varnames = list(raw.columns[1:])

# move the 6th column (INDPRO) to the first column
perm = [5] + list(range(5)) + list(range(6, data.shape[1]))
data = data[:, perm]
varnames = [varnames[j] for j in perm]

# remove columns with missing values
idx = ~np.isnan(data).any(axis=0)
data = data[:, idx]
varnames = [v for v, keep in zip(varnames, idx) if keep]

# standardize the data
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0, ddof=1)
Y = (data - data_mean)/data_std

T, n = Y.shape

# storage
store_F = np.zeros((nsim, T, r))
store_A = np.zeros((nsim, n, r))
store_sig2 = np.zeros((nsim, n))
store_omega2 = np.zeros((nsim, r))
store_phi = np.zeros((nsim, r))

# prior hyperparameters
a0 = 0
Va = 1   # a_ij iid N(a0,Va)
phi0 = np.zeros(r)
Vphi = np.ones(r)
nusig2 = 3
Ssig2 = (nusig2-1)*np.ones(n)
nuomega2 = 3
Somega2 = (nuomega2-1)*np.ones(r)

# initialize the Markov chain
sig2 = np.var(Y, axis=0, ddof=1)
omega2 = np.ones(r)
phi = 0.5*np.ones(r)
Phi = sp.diags(phi, 0, shape=(r, r), format='csc')
A = np.vstack((np.eye(r), np.zeros((n-r, r))))   # lower-triangular normalization

# matrices used to build H_phi
hzeros = sp.csc_array((r, (T-1)*r))
vzeros = sp.csc_array((T*r, r))
HPhi = sp.identity(T*r, format='csc') \
    - sp.hstack([sp.vstack([hzeros, sp.kron(sp.identity(T-1), Phi)]), vzeros],
                format='csc')

for isim in range(nsim + burnin):
    # sample f
    # Kf is banded with bandwidth r; factor and solve it in banded storage
    iOmega = sp.diags(np.tile(1/omega2, T), 0, format='csc')
    AiSigA = A.T @ (A*(1/sig2)[:, None])   # A' * iSig * A, r x r
    Kf = HPhi.T @ iOmega @ HPhi + sp.kron(sp.identity(T, format='csc'), AiSigA)
    ab = np.zeros((r+1, T*r))              # upper banded storage of Kf
    for i in range(r+1):
        ab[r-i, i:] = Kf.diagonal(i)
    # kron(I_T, A'*iSig)*y stacks A'*iSig*y_t over t
    bf = ((Y*(1/sig2)) @ A).flatten()
    f_hat = solveh_banded(ab, bf)
    U = cholesky_banded(ab)                # Kf = U'U with U upper banded
    f = f_hat + solve_banded((0, r), U, np.random.randn(T*r))
    F = f.reshape(T, r)                    # T x r - tth row is f_t

    # sample A equation by equation
    for i in range(1, n):
        nai = min(i, r)
        Xf = F[:, :nai]
        K_ai = np.eye(nai)/Va + (Xf.T @ Xf)/sig2[i]
        if i < r:
            ai_hat = np.linalg.solve(
                K_ai, (a0/Va)*np.ones(nai) + Xf.T @ (Y[:, i] - F[:, i])/sig2[i])
        else:
            ai_hat = np.linalg.solve(
                K_ai, (a0/Va)*np.ones(nai) + Xf.T @ Y[:, i]/sig2[i])
        A[i, :nai] = ai_hat + solve_triangular(np.linalg.cholesky(K_ai).T,
                                               np.random.randn(nai), lower=False)

    # sample sig2
    E_y = Y - F @ A.T
    sig2 = 1/np.random.gamma(nusig2 + T/2, 1/(Ssig2 + np.sum(E_y**2, axis=0)/2))

    # sample omega2
    E_f = np.vstack((F[:1, :], F[1:, :] - F[:-1, :] @ np.diag(phi)))
    omega2 = 1/np.random.gamma(nuomega2 + T/2,
                               1/(Somega2 + np.sum(E_f**2, axis=0)/2))

    # sample phi equation by equation (normal truncated to |phi|<1)
    Zf = np.vstack((np.zeros((1, r)), F[:-1, :]))
    for jj in range(r):
        Kphi_j = 1/Vphi[jj] + np.sum(Zf[:, jj]**2)/omega2[jj]
        phi_hat_j = (phi0[jj]/Vphi[jj]
                     + np.sum(Zf[:, jj]*F[:, jj])/omega2[jj])/Kphi_j
        phi_sd_j = np.sqrt(1/Kphi_j)
        accepted = False
        while not accepted:
            phi_prop = phi_hat_j + phi_sd_j*np.random.randn()
            if abs(phi_prop) < 1:
                phi[jj] = phi_prop
                accepted = True
    Phi = sp.diags(phi, 0, shape=(r, r), format='csc')
    HPhi = sp.identity(T*r, format='csc') \
        - sp.hstack([sp.vstack([hzeros, sp.kron(sp.identity(T-1), Phi)]),
                     vzeros], format='csc')

    if isim >= burnin:
        isave = isim - burnin
        store_F[isave, :, :] = F
        store_A[isave, :, :] = A
        store_sig2[isave, :] = sig2
        store_omega2[isave, :] = omega2
        store_phi[isave, :] = phi

    if (isim+1) % 5000 == 0:
        print('Iteration %d of %d (%.1f%%)'
              % (isim+1, nsim + burnin, 100*(isim+1)/(nsim + burnin)))

F_mean = np.mean(store_F, axis=0)

# posterior summary of the AR coefficient
phi_mean = np.mean(store_phi, axis=0)
phi_q = np.quantile(store_phi, [0.025, 0.975], axis=0)
print('Posterior mean of phi = %.3f, 95%% CI = (%.3f, %.3f)'
      % (phi_mean[0], phi_q[0, 0], phi_q[1, 0]))

# decimal-year time axis for plotting
tt = year_part + (month_part - 1)/12

# full-sample and pre-COVID plots of the posterior mean factor
idx_pre = tt <= 2019 + 11/12   # through December 2019
tt_pre_end = tt[idx_pre][-1]

plt.figure(figsize=(9, 6))
# top panel: full sample
plt.subplot(2, 1, 1)
yl_full = (np.min(F_mean[:, 0]) - 0.2, np.max(F_mean[:, 0]) + 0.2)
plt.plot(tt, F_mean[:, 0], 'k', linewidth=1.5)
plt.axhline(0, color='k', linewidth=0.8)
plt.xlim(tt[0], tt[-1])
plt.ylim(yl_full)
shade_nber_recessions(yl_full[0], yl_full[1])

# bottom panel: pre-COVID sample
plt.subplot(2, 1, 2)
yl_pre = (np.min(F_mean[idx_pre, 0]) - 0.2, np.max(F_mean[idx_pre, 0]) + 0.2)
plt.plot(tt[idx_pre], F_mean[idx_pre, 0], 'k', linewidth=1.5)
plt.axhline(0, color='k', linewidth=0.8)
plt.xlim(tt[0], tt_pre_end)
plt.ylim(yl_pre)
shade_nber_recessions(yl_pre[0], yl_pre[1])

plt.tight_layout()
plt.savefig('DFM_f.eps')
plt.show()
