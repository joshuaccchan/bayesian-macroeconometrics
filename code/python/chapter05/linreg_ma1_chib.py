# linreg_ma1_chib.py
# Computes the log marginal likelihood of the ARMA(2,1) model for US
# PCE inflation via Chib's method. The posterior ordinate at the
# posterior mean (beta*, sig2*, psi*) is factored as
#   p(beta*, sig2*, psi* | y) = p(beta* | y) * p(sig2* | y, beta*)
#                               * p(psi* | y, beta*, sig2*).
# The first factor is estimated from the main Gibbs run, the second
# from a reduced run that fixes beta at beta* and updates (sig2, psi),
# and the third by evaluating the conditional posterior of psi on a
# fine grid. The seed is reset before the reduced run so the marginal
# likelihood estimate is reproducible.

import numpy as np
from scipy.linalg import solve_banded
from loglike_MA1 import loglike_MA1
from sample_psi_RW import sample_psi_RW
from lmvnpdf import lmvnpdf
from ligampdf import ligampdf

exec(open('linreg_ma1.py').read())  # estimate the ARMA(2,1) model

nsim_re = 5000  # size of reduced runs
nsim, m = store_theta.shape
T = y.size

theta_hat = store_theta.mean(axis=0)
beta_s = theta_hat[:m-2]  # beta*
sig2_s = theta_hat[m-2]   # sig2*
psi_s = theta_hat[m-1]    # psi*

# log likelihood at theta*
llike = loglike_MA1(psi_s, y - X @ beta_s, sig2_s)

# log prior at theta*
log_prior = lambda b, s, p: lmvnpdf(b, beta0, np.linalg.inv(iVbeta)) \
    + ligampdf(s, nu0, S0) + np.log(1/2)

# 1) posterior of beta at beta_s using main run
store_lpbeta = np.zeros(nsim)
for isim in range(nsim):
    sig2 = store_theta[isim, m-2]  # from the main run
    psi = store_theta[isim, m-1]
    Hpsi = np.vstack((np.ones(T), np.full(T, psi)))
    X_tilde = solve_banded((1, 0), Hpsi, X)
    y_tilde = solve_banded((1, 0), Hpsi, y)
    Dbeta = np.linalg.inv(iVbeta + X_tilde.T @ X_tilde/sig2)
    beta_hat = Dbeta @ (iVbeta @ beta0
        + X_tilde.T @ y_tilde/sig2)
    store_lpbeta[isim] = \
        lmvnpdf(beta_s, beta_hat, Dbeta)
a = store_lpbeta.max()
lpbeta = np.log(np.mean(np.exp(store_lpbeta - a))) + a

# 2) posterior of sig2 at sig2_s using a reduced run
np.random.seed(42)  # re-seed for reproducible reduced run
beta = beta_s  # fix beta at beta_s
e = y - X @ beta
sig2 = sig2_s
psi = psi_s
store_lpsig2 = np.zeros(nsim_re)
for isim in range(nsim_re):
    # sample psi
    psi, _ = sample_psi_RW(psi, e, sig2, g_var)
    Hpsi = np.vstack((np.ones(T), np.full(T, psi)))
    # sample sig2
    tmp = solve_banded((1, 0), Hpsi, e)
    sig2 = 1/np.random.gamma(nu0 + T/2, 1/(S0 + tmp @ tmp/2))

    store_lpsig2[isim] = \
        ligampdf(sig2_s, nu0 + T/2, S0 + tmp @ tmp/2)
a = store_lpsig2.max()
lpsig2 = np.log(np.mean(np.exp(store_lpsig2 - a))) + a

# 3) posterior of psi at psi_s via grid
n_grid = 399
psi_grid = np.linspace(-.99, .99, n_grid)
# ensure psi_s is on the grid
psi_grid = np.sort(np.append(psi_grid, psi_s))
n_grid = psi_grid.size
idx_psi = (psi_grid == psi_s)  # index for psi_s
lp_psi = np.zeros(n_grid)  # log posterior density
for igrid in range(n_grid):
    psi = psi_grid[igrid]
    lp_psi[igrid] = loglike_MA1(psi, y - X @ beta_s, sig2_s)
p_psi = np.exp(lp_psi - lp_psi.max())
p_psi = p_psi/np.trapezoid(p_psi, psi_grid)
lppsi = np.log(p_psi[idx_psi][0])

log_ml = llike + log_prior(beta_s, sig2_s, psi_s) \
    - (lpbeta + lpsig2 + lppsi)

print(f'Log marginal likelihood: {log_ml:.2f}')
