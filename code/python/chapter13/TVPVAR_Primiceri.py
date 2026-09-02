"""TVPVAR_Primiceri.py
Application of the TVP-VAR with stochastic volatility to U.S. GDP
deflator inflation, unemployment, and the 3-month T-bill rate, replicating
the dataset and sample of Primiceri (2005).

Requires: estimate_TVPVAR.py, construct_IR.py, plotCI.py, SURform.py, SVRW.py
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular

from estimate_TVPVAR import estimate_TVPVAR
from construct_IR import construct_IR
from plotCI import plotCI

np.random.seed(42)

p = 2
nsim = 20000
burnin = 5000
n_hz = 21  # impulse response horizon: 0 to 20 quarters

# load data: 1953Q1-2001Q3, matching Primiceri's sample.
# macro_Primiceri_Q.csv is built by build_macro_Primiceri.m from FRED
# series GDPDEF, UNRATE, TB3MS; columns B:D are INFL, UNRATE, TB3MS.
data = pd.read_csv('macro_Primiceri_Q.csv').iloc[:, 1:4].to_numpy()
Y_all = data[:195, :]                    # 1953Q1-2001Q3
Tall, n = Y_all.shape
k = 1 + n*p
nk = n*k
m = n*(n-1)//2

# split: first 40 obs for prior calibration, rest for inference
T_train = 40
Y_train = Y_all[:T_train, :]             # 1953Q1-1962Q4
Y_est = Y_all[T_train-p:, :]             # 1962Q3-2001Q3 (carries p lags)
T_eff = Tall - T_train                   # 155 effective obs (1963Q1-2001Q3)
print(f'Sample: {Tall} obs total, {T_train} for training, T_eff = {T_eff}.')

# 0-based (row, col) positions of the free entries below the diagonal of L,
# eq-by-eq order (replaces the MATLAB linear-index vector L_id)
r_pos = np.array([i for i in range(1, n) for _ in range(i)])
c_pos = np.array([j for i in range(1, n) for j in range(i)])

# training-sample OLS for prior centers and scales
T_tr_eff = T_train - p
X_tilde_tr = np.zeros((T_tr_eff, n*p))
for i in range(1, p + 1):
    X_tilde_tr[:, (i-1)*n:i*n] = Y_train[p-i:T_train-i, :]
Z_tr = np.hstack((np.ones((T_tr_eff, 1)), X_tilde_tr))
A_tr = np.linalg.lstsq(Z_tr, Y_train[p:, :], rcond=None)[0]
E_tr = Y_train[p:, :] - Z_tr @ A_tr
Sig_tr = E_tr.T @ E_tr/T_tr_eff
beta_ols = A_tr.T.flatten(order='F')     # reshape(A_tr', nk, 1)
Vb_ols = np.kron(Sig_tr, np.linalg.inv(Z_tr.T @ Z_tr))

# l_OLS from modified-Cholesky Sig_tr = L^{-1} D (L')^{-1}
Lc_tr = np.linalg.cholesky(Sig_tr)
L_ols = np.diag(np.diag(Lc_tr)) @ np.linalg.inv(Lc_tr)
L_ols = np.tril(L_ols)
l_ols = L_ols[r_pos, c_pos]

# per-equation sampling-variance blocks for l_OLS
sig2_full = np.diag(Lc_tr)**2
Vl_blocks = []
for ii in range(1, n):
    Ein = -E_tr[:, :ii]
    Vl_blocks.append(sig2_full[ii]*np.linalg.inv(Ein.T @ Ein))
Vl_full = np.zeros((m, m))
off = 0
for ii in range(1, n):
    Vl_full[off:off+ii, off:off+ii] = Vl_blocks[ii-1]
    off += ii

# Prior calibration:
kQ = 0.01
kS = 0.1
prior = {}
prior['beta0_mean'] = beta_ols
prior['beta0_var'] = 100*np.eye(nk)      # diffuse prior on beta_0
prior['l0_mean'] = l_ols
prior['l0_var'] = 100*np.eye(m)          # diffuse prior on l_0
prior['h0_mean'] = np.log(np.diag(Lc_tr)**2)
prior['h0_var'] = 100*np.eye(n)          # diffuse prior on h_0
prior['Q_nu'] = T_train
prior['Q_S0'] = kQ**2*T_train*Vb_ols
prior['S_nu'] = np.arange(1, n) + 1
prior['S_S0'] = [kS**2*(ii+1)*Vl_blocks[ii-1] for ii in range(1, n)]
prior['sigh2_nu'] = 1*np.ones(n)
prior['sigh2_S0'] = 0.5*np.ones(n)

# run the Gibbs sampler
print(f'Running Gibbs sampler for TVP-VAR (nsim={nsim}, burnin={burnin}) ...')
start = time.time()
store_beta, store_l, store_h, _, store_Q = \
    estimate_TVPVAR(Y_est, p, prior, nsim, burnin)[:5]
print(f'Elapsed time is {time.time() - start:.1f} seconds.')
Q_mean = store_Q.mean(axis=0).reshape(nk, nk, order='F')
print(f'\nQ posterior mean diag stats: min={np.min(np.diag(Q_mean)):.3e} '
      f'median={np.median(np.diag(Q_mean)):.3e} '
      f'max={np.max(np.diag(Q_mean)):.3e}')

dates_q = 1963 + 0.25*np.arange(T_eff)

# residual standard deviations: sqrt(diag(Sigma_t)) at each draw
l_3d = store_l.reshape(nsim, T_eff, m)   # element (s, t, j) = l_t[j] at draw s
l21 = l_3d[:, :, 0]
l31 = l_3d[:, :, 1]
l32 = l_3d[:, :, 2]
d1 = np.exp(store_h[:, :, 0])
d2 = np.exp(store_h[:, :, 1])
d3 = np.exp(store_h[:, :, 2])

sig2_red = np.zeros((nsim, T_eff, n))
sig2_red[:, :, 0] = d1
sig2_red[:, :, 1] = l21**2*d1 + d2
sig2_red[:, :, 2] = (l32*l21 - l31)**2*d1 + l32**2*d2 + d3
sig_red = np.sqrt(sig2_red)

sig_med = np.quantile(sig_red, 0.50, axis=0)
sig_lo = np.quantile(sig_red, 0.16, axis=0)
sig_hi = np.quantile(sig_red, 0.84, axis=0)

# Figure 1: residual standard deviations over time
resnames = ['Inflation', 'Unemployment', '3-month T-bill']
plt.figure(figsize=(7, 5))
for ii in range(n):
    plt.subplot(n, 1, ii + 1)
    plotCI(dates_q, sig_lo[:, ii], sig_hi[:, ii])
    plt.plot(dates_q, sig_med[:, ii], 'k', linewidth=1.5)
    plt.xlim(dates_q[0], dates_q[-1])
    if ii == n - 1:
        plt.xlabel('Year')
plt.tight_layout()
plt.savefig('Primiceri_sigma.eps')

# MP shock IRFs at three reference dates
ref_dates = np.array([1975, 1981.5, 1996])   # 1975Q1, 1981Q3, 1996Q1
ref_labels = ['1975Q1', '1981Q3', '1996Q1']
n_ref = len(ref_dates)
i_mp = n - 1                                 # MP shock = last equation
shock = np.zeros(n)
shock[i_mp] = 1
F_sub = np.hstack((np.eye((p-1)*n), np.zeros(((p-1)*n, n))))

store_yIR = np.zeros((nsim, n_hz, n, n_ref))
maxeig_all = np.zeros((nsim, n_ref))
for r in range(n_ref):
    t_idx = np.argmin(np.abs(dates_q - ref_dates[r]))
    for s in range(nsim):
        beta_path = store_beta[s, :].reshape(nk, T_eff, order='F')
        beta_t = beta_path[:, t_idx]
        A_full = beta_t.reshape(k, n, order='F').T
        F = np.vstack((A_full[:, 1:], F_sub))
        maxeig_all[s, r] = np.max(np.abs(np.linalg.eigvals(F)))
        l_path = store_l[s, :].reshape(m, T_eff, order='F')
        l_t = l_path[:, t_idx]
        h_t = store_h[s, t_idx, :]
        Lt = np.eye(n)
        Lt[r_pos, c_pos] = l_t
        Dt = np.diag(np.exp(h_t))
        # Sig_t = Lt \ Dt / Lt'
        Sig_t = solve_triangular(Lt, solve_triangular(Lt, Dt, lower=True).T,
                                 lower=True).T
        store_yIR[s, :, :, r] = construct_IR(beta_t, Sig_t, n_hz, shock)
    print('Reference %s -> t = %3d : max|eig(F)| median=%.3f  q10=%.3f  '
          'q90=%.3f  max=%.3f' % (
              ref_labels[r], t_idx,
              np.quantile(maxeig_all[:, r], 0.5),
              np.quantile(maxeig_all[:, r], 0.1),
              np.quantile(maxeig_all[:, r], 0.9),
              np.max(maxeig_all[:, r])))

# normalize by posterior-median impact FFR response (one scalar per ref date)
# so the IRF corresponds to a 1pp shock; dividing each draw by its own
# impact response amplifies noise from draws with near-zero impact
for r in range(n_ref):
    impact_med = np.quantile(store_yIR[:, 0, i_mp, r], 0.50)
    store_yIR[:, :, :, r] = store_yIR[:, :, :, r]/impact_med

yIR_med = np.quantile(store_yIR, 0.50, axis=0)
yIR_lo = np.quantile(store_yIR, 0.16, axis=0)
yIR_hi = np.quantile(store_yIR, 0.84, axis=0)

# Figure 2: inflation and unemployment IRFs, shared y-axis per row
varnames_resp = ['Inflation', 'Unemployment']
n_resp = len(varnames_resp)
hz = np.arange(n_hz)

y_lim = np.zeros((n_resp, 2))
for ii in range(n_resp):
    y_min = np.min(yIR_lo[:, ii, :])
    y_max = np.max(yIR_hi[:, ii, :])
    pad = 0.05*(y_max - y_min)
    y_lim[ii, :] = [y_min - pad, y_max + pad]

plt.figure(figsize=(8, 3.5))
for ii in range(n_resp):
    for r in range(n_ref):
        plt.subplot(n_resp, n_ref, ii*n_ref + r + 1)
        plotCI(hz, yIR_lo[:, ii, r], yIR_hi[:, ii, r])
        plt.plot(hz, yIR_med[:, ii, r], 'k', linewidth=1.5)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.xlim(-0.5, n_hz - 1)
        plt.ylim(y_lim[ii, 0], y_lim[ii, 1])
        if r == 0:
            plt.ylabel(varnames_resp[ii])
        if ii == n_resp - 1:
            plt.xlabel('Quarters')
plt.tight_layout()
plt.savefig('Primiceri_MPshock_IR.eps')

# save plot data (Primiceri_results.npz) so the figures can be rebuilt
# without rerunning
np.savez('Primiceri_results.npz',
         sig_med=sig_med, sig_lo=sig_lo, sig_hi=sig_hi, dates_q=dates_q,
         yIR_med=yIR_med, yIR_lo=yIR_lo, yIR_hi=yIR_hi, hz=hz,
         ref_labels=ref_labels, i_mp=i_mp,
         n=n, p=p, k=k, nk=nk, m=m, T_eff=T_eff)

# summary of final results
print('\nPosterior median residual std devs (1963Q1 / 2001Q3):')
for ii in range(n):
    print(f'  {resnames[ii]:<16s}: {sig_med[0, ii]:.3f} / {sig_med[-1, ii]:.3f}')
print('\nMedian [16%, 84%] responses to a 1pp MP shock at horizon 8:')
for ii in range(n_resp):
    for r in range(n_ref):
        print(f'  {varnames_resp[ii]:<12s} {ref_labels[r]}: '
              f'{yIR_med[8, ii, r]: .3f} '
              f'[{yIR_lo[8, ii, r]: .3f}, {yIR_hi[8, ii, r]: .3f}]')

plt.show()
