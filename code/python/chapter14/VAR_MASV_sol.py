"""VAR_MASV_sol.py
Solution to the exercise "Moving Average Stochastic Volatility", part (c):
a Bayesian VAR whose reduced-form errors follow a common MA(1) with a common
stochastic-volatility innovation:
    y_t   = A'x_t + eps_t,
    eps_t = v_t + psi_1 v_{t-1},          |psi_1| < 1,
    (v_t | Sig,h_t) ~ N(0, exp(h_t) Sig),
    h_t   = phi h_{t-1} + u_t^h,   u_t^h ~ N(0,sigh2),  h_1 ~ N(0,sigh2/(1-phi^2)).

Stacking the T observations (v_0 = 0) gives U = M_psi V, with M_psi the T x T
unit lower-bidiagonal MA operator (1 on the diagonal, psi_1 on the first
subdiagonal) and V the T x n whitened innovations (rows v_t'). Hence
vec(U) ~ N(0, Sig kron Omega) with Omega = M_psi diag(exp(h)) M_psi' the
symmetric tridiagonal (banded) matrix with
    Omega_11   = exp(h_1),
    Omega_tt   = exp(h_t) + psi_1^2 exp(h_{t-1}),  t >= 2,
    Omega_{t,t-1} = psi_1 exp(h_{t-1}).
Conditional on Omega, (A,Sig) has the natural-conjugate NIW posterior of
Theorem 15.1 (thm:largeVAR-cond-NIW). All T x T operations use the banded
Omega and its Cholesky -- no dense T x T inverses are formed.

Gibbs blocks:
  1. (A,Sig | Y,Omega): natural-conjugate NIW.
  2. (psi_1 | Y,A,Sig,h): random-walk Metropolis-Hastings on (-1,1). Because
     |M_psi|^n = 1, the log-conditional is
     -0.5 sum_t exp(-h_t) v_t(psi_1)' Sig^{-1} v_t(psi_1) + log prior(psi_1),
     with V(psi_1) = M_psi \\ U obtained by a bidiagonal solve.
  3. (h | Y,A,Sig,psi_1,phi,sigh2): whiten V = M_psi \\ U so v_t ~ N(0,e^{h_t}Sig)
     is a common-SV structure, then the same ARMH update as VAR_CSV_o.py.
  4. (phi,sigh2 | h): exactly as in VAR_CSV_o.py.
The outlier component of VAR_CSV_o.py is omitted here for simplicity.

Requires Minn_NCP.py, sample_CSV_h_ARMH.py, SV_RW_gaussian_approx.py,
shaded_band.py, and inefficiency_factor.py (with specvar0.py).
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cho_solve, solve_triangular, solveh_banded, solve_banded
from scipy.stats import invwishart

from Minn_NCP import Minn_NCP
from sample_CSV_h_ARMH import sample_CSV_h_ARMH
from SV_RW_gaussian_approx import SV_RW_gaussian_approx
from shaded_band import shaded_band
from inefficiency_factor import inefficiency_factor

np.random.seed(42)

p = 12             # number of lags (monthly data)
nsims = 10000
burnin = 2000
sig_psi = 0.03     # random-walk MH proposal std for psi_1

# ---- load FRED-MD data (same 25-variable panel as VAR_CSV_o.py) ----
raw = pd.read_csv('FRED-MD.csv')
vars_ = ['RPI', 'DPCERA3M086SBEA', 'INDPRO', 'CUMFNS', 'UNRATE', 'PAYEMS',
         'CES0600000007', 'CES0600000008', 'WPSFD49207', 'PCEPI', 'HOUST',
         'S&P 500', 'EXUSUKx', 'GS5', 'GS10', 'BAAFFM',   # Carriero et al. (2024)
         'FEDFUNDS', 'TB3MS', 'GS1', 'M2SL', 'BUSLOANS', 'CPIAUCSL',
         'OILPRICEx', 'RETAILx', 'PERMIT']                # additional series
data = raw[vars_].to_numpy()
dates = raw.iloc[:, 0].to_numpy()
ok = ~np.isnan(data).any(axis=1)
data = data[ok, :]
dates = dates[ok]
Y0 = data[:p, :]
Y = data[p:, :]
dates = dates[p:]
T, n = Y.shape
k = n*p + 1
print('FRED-MD: n=%d variables, sample %.2f-%.2f (%d months)'
      % (len(vars_), dates[0], dates[-1], T))

# ---- prior hyperparameters ----
kappa1 = 100
kappa2 = 0.2**2
rw = 0
A0, VA, nu0, S0 = Minn_NCP(Y, Y0, p, kappa1, kappa2, rw)
iva = 1/np.diag(VA)   # diagonal of VA^{-1}
phi0 = 0.95
Vphi = 0.1**2                  # common SV: phi ~ N(phi0,Vphi) on (-1,1)
nuh0 = 20
Sh0 = 0.05*(nuh0 - 1)          # sigh2 ~ IG(nuh0,Sh0)
# psi_1 ~ U(-1,1): flat prior, so it drops out of the MH ratio

# ---- regressor matrix Z = [1, y_{t-1}',...,y_{t-p}'] ----
tmpY = np.vstack((Y0[-p:, :], Y))
Z = np.zeros((T, n*p))
for i in range(1, p + 1):
    Z[:, (i-1)*n:i*n] = tmpY[p-i:len(tmpY)-i, :]
Z = np.column_stack((np.ones(T), Z))

# ---- storage ----
store_A = np.zeros((k, n))
store_Sig = np.zeros((n, n))
store_h = np.zeros((nsims, T))
store_psi = np.zeros(nsims)
store_theta = np.zeros((nsims, 2))   # [phi, sigh2]
counth = 0
countphi = 0
countpsi = 0

# ---- initialize the Markov chain ----
phi = phi0
sigh2 = 0.1
psi_1 = 0
A_ols = np.linalg.solve(Z.T @ Z + np.diag(iva), Z.T @ Y)
U_ols = Y - Z @ A_ols
C_ols = np.linalg.cholesky(U_ols.T @ U_ols/T)
s2_init = np.sum(solve_triangular(C_ols, U_ols.T, lower=True).T**2, axis=1)
h = SV_RW_gaussian_approx(s2_init, np.mean(np.log(s2_init)), sigh2)
h = h - np.mean(h)

# banded storage for the tridiagonal Omega (bandwidth 1) and the unit
# lower-bidiagonal MA operator M_psi, rebuilt each sweep
ab_Om = np.zeros((2, T))   # upper banded form: row 0 superdiag, row 1 diag
ab_M = np.zeros((2, T))    # (l,u) = (1,0) form: row 0 diag, row 1 subdiag
ab_M[0, :] = 1

print('Starting MCMC for the VAR with common MA(1)-SV errors....')
start_time = time.time()
for isim in range(nsims + burnin):
    eh = np.exp(h)

    # ---- banded tridiagonal Omega = M_psi diag(exp(h)) M_psi' ----
    md = eh.copy()
    md[1:] = eh[1:] + psi_1**2*eh[:-1]      # main diagonal
    od = psi_1*eh[:-1]                      # first off-diagonal
    ab_Om[1, :] = md
    ab_Om[0, 1:] = od
    OmiZ = solveh_banded(ab_Om, Z)          # Omega^{-1} Z via banded solves
    OmiY = solveh_banded(ab_Om, Y)          # Omega^{-1} Y via banded solves

    # ---- block 1: (A,Sig | Y,Omega) natural-conjugate NIW ----
    KA = np.diag(iva) + Z.T @ OmiZ
    KA = (KA + KA.T)/2
    CKA = np.linalg.cholesky(KA)
    Ahat = cho_solve((CKA, True), iva[:, None]*A0 + Z.T @ OmiY)
    Shat = (S0 + A0.T @ (iva[:, None]*A0) + Y.T @ OmiY - Ahat.T @ KA @ Ahat)
    Shat = (Shat + Shat.T)/2
    Sig = invwishart.rvs(df=nu0 + T, scale=Shat)
    CSig = np.linalg.cholesky(Sig)
    A = Ahat + solve_triangular(CKA.T, np.random.randn(k, n),
                                lower=False) @ CSig.T

    U = Y - Z @ A   # reduced-form residuals (rows eps_t')

    # ---- block 2: psi_1 via random-walk MH on (-1,1) (flat prior) ----
    ab_M[1, :-1] = psi_1
    V = solve_banded((1, 0), ab_M, U)   # whitened innovations, rows v_t'
    W = solve_triangular(CSig, V.T, lower=True).T  # v_t standardized
    llik = -0.5*np.sum(np.exp(-h)*np.sum(W**2, axis=1))
    psic = psi_1 + sig_psi*np.random.randn()
    if np.abs(psic) < 1:
        ab_M[1, :-1] = psic
        Vc = solve_banded((1, 0), ab_M, U)
        Wc = solve_triangular(CSig, Vc.T, lower=True).T
        llikc = -0.5*np.sum(np.exp(-h)*np.sum(Wc**2, axis=1))
        if np.log(np.random.rand()) < llikc - llik:
            psi_1 = psic
            countpsi += 1
            W = Wc

    # ---- block 3: common log-volatility h via the Laplace-based ARMH ----
    s2 = np.sum(W**2, axis=1)   # per-period sum of squares over n vars
    h, flag = sample_CSV_h_ARMH(s2, phi, sigh2, h, n, 30)
    counth += flag
    h = h - np.mean(h)          # common SV normalized to zero mean

    # ---- block 4: sigh2 and phi (exactly as in VAR_CSV_o.py) ----
    eh_ar = np.concatenate(([h[0]*np.sqrt(1 - phi**2)], h[1:] - phi*h[:-1]))
    # np.random.gamma uses (shape, scale)
    sigh2 = 1/np.random.gamma(nuh0 + T/2, 1/(Sh0 + np.sum(eh_ar**2)/2))
    Kphi = 1/Vphi + np.sum(h[:T-1]**2)/sigh2
    phihat = (phi0/Vphi + h[:T-1] @ h[1:T]/sigh2)/Kphi
    phic = phihat + np.random.randn()/np.sqrt(Kphi)

    def gphi(x):
        return -.5*np.log(sigh2/(1 - x**2)) - .5*(1 - x**2)/sigh2*h[0]**2
    if np.abs(phic) < .9999:
        if np.exp(gphi(phic) - gphi(phi)) > np.random.rand():
            phi = phic
            countphi += 1

    if isim >= burnin:
        isave = isim - burnin
        store_A += A
        store_Sig += Sig
        store_h[isave, :] = h
        store_psi[isave] = psi_1
        store_theta[isave, :] = [phi, sigh2]

    if (isim + 1) % 1000 == 0:
        print('%d loops... ' % (isim + 1))
print('MCMC takes %.1f seconds' % (time.time() - start_time))

# ---- posterior summaries ----
store_h = store_h - store_h.mean(axis=1, keepdims=True)  # level of h not
                                                         # separately identified
A_mean = store_A/nsims
Sig_mean = store_Sig/nsims
theta_mean = store_theta.mean(axis=0)
h_mean = store_h.mean(axis=0)
vol = np.exp(store_h/2)                 # e^{h_t/2}
vol_mean = vol.mean(axis=0)
vol_ci = np.quantile(vol, [.05, .95], axis=0).T

psi_mean = np.mean(store_psi)
psi_ci = np.quantile(store_psi, [.05, .95])
print('psi_1 posterior mean %.4f, 90%% CI [%.4f, %.4f]'
      % (psi_mean, psi_ci[0], psi_ci[1]))
print('psi_1 MH acceptance rate: %.3f' % (countpsi/(nsims + burnin)))
print('acceptance rate (h): %.3f ; (phi): %.3f'
      % (counth/(nsims + burnin), countphi/(nsims + burnin)))
print('phi posterior mean %.3f ; sigh2 posterior mean %.4f'
      % (theta_mean[0], theta_mean[1]))

IF_psi = inefficiency_factor(store_psi, 500)[0]
IF_h = inefficiency_factor(store_h, 500)
print('IF of psi_1: %.1f ; IF of h: mean %.1f, median %.1f'
      % (IF_psi, np.mean(IF_h), np.median(IF_h)))

# ---- figures (saved as vector PDFs to the figures folder) ----
figdir = 'figures'  # output folder (was an absolute path in the author's setup)
os.makedirs(figdir, exist_ok=True)

# (i) common volatility e^{h_t/2} with 90% credible band
fig1 = plt.figure(figsize=(8, 3))
shaded_band(dates, vol_ci[:, 0], vol_ci[:, 1])
plt.plot(dates, vol_mean, 'k')
plt.tight_layout()
fig1.savefig(os.path.join(figdir, 'sol_largeVAR_MASV_h.pdf'))

# (ii) histogram of the posterior draws of psi_1
fig2 = plt.figure(figsize=(5, 3.5))
plt.hist(store_psi, 50, density=True, color=(.7, .7, .7))
plt.xlabel(r'$\psi_1$')
plt.tight_layout()
fig2.savefig(os.path.join(figdir, 'sol_largeVAR_MASV_psi.pdf'))

print('figures written to %s' % figdir)
plt.show()
