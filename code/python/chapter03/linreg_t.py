# linreg_t.py
# Gibbs sampler for an AR(2) model of US PCE inflation with Student-t
# errors. The model is
#   y_t = beta_1 + beta_2 y_{t-1} + beta_3 y_{t-2} + eps_t,
#   (eps_t | lam_t) ~ N(0, sig2*lam_t),  lam_t ~ IG(nu/2, nu/2),
# with independent normal-inverse-gamma priors on (beta, sig2) and a
# uniform prior on nu over (2, nu_ub). The block for nu uses a
# Griddy-Gibbs step via sample_nu_griddy.py.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sample_nu_griddy import sample_nu_griddy
from shaded_band import shaded_band

np.random.seed(42)
nsim = 20000
burnin = 1000

# load data (column PCECTPI; rows 1960Q1-2019Q4)
data = pd.read_csv('USPCE.csv')['PCECTPI'].to_numpy()[:240]
y0 = data[:4]   # initial conditions: [y_{-3},y_{-2},y_{-1},y_0]
y = data[4:]    # sample used for estimation
T = y.size

# regressors for AR(2): [1, y_{t-1}, y_{t-2}]
xlag1 = np.concatenate(([y0[3]], y[:-1]))
xlag2 = np.concatenate((y0[2:4], y[:-2]))
X = np.column_stack((np.ones(T), xlag1, xlag2))
k = X.shape[1]   # number of regressors

# prior hyperparameters (indep normal and inverse-gamma)
beta0 = np.zeros(k)
iVbeta = np.eye(k)/100   # prior precision of beta
nu0 = 4
S0 = 1
nu_ub = 50               # prior upperbound of nu

# initialize the Markov chain
nu = 5
beta = np.linalg.solve(X.T @ X, X.T @ y)
sig2 = np.sum((y - X @ beta)**2)/T
lam = np.ones(T)
ilam = 1/lam   # iLam is diagonal; keep 1/lam as a vector

store_theta = np.zeros((nsim, k+2))   # [beta', sig2, nu]
store_lam = np.zeros((nsim, T))
n_grid = 500

for isim in range(nsim + burnin):
    # sample beta
    Dbeta = np.linalg.solve(iVbeta + (X*ilam[:, None]).T @ X/sig2, np.eye(k))
    beta_hat = Dbeta @ (iVbeta @ beta0 + X.T @ (ilam*y)/sig2)
    C = np.linalg.cholesky(Dbeta)
    beta = beta_hat + C @ np.random.randn(k)

    # sample sig2
    e = y - X @ beta
    sig2 = 1/np.random.gamma(nu0 + T/2, 1/(S0 + e @ (ilam*e)/2))

    # sample lam
    lam = 1/np.random.gamma((nu+1)/2, 2/(nu + e**2/sig2))
    ilam = 1/lam

    # sample nu
    nu = sample_nu_griddy(lam, nu_ub, n_grid)

    # store the parameters
    if isim >= burnin:
        isave = isim - burnin
        store_theta[isave, :] = np.concatenate((beta, [sig2, nu]))
        store_lam[isave, :] = lam

# Posterior summaries
theta_mean = store_theta.mean(axis=0)
theta_lo = np.quantile(store_theta, .025, axis=0)
theta_hi = np.quantile(store_theta, .975, axis=0)

lam_mean = store_lam.mean(axis=0)
lam_lo = np.quantile(store_lam, 0.025, axis=0)
lam_hi = np.quantile(store_lam, 0.975, axis=0)

print("Posterior mean of [beta', sig2, nu]:")
print(theta_mean)
print('95% posterior intervals (rows: 2.5%, 97.5%):')
print(np.vstack((theta_lo, theta_hi)))

plt.figure(figsize=(10, 3.75))

# left panel: posterior of nu
plt.subplot(1, 2, 1)
plt.hist(store_theta[:, -1], 30, facecolor=(0.8, 0.8, 0.8), edgecolor='k')
plt.xlabel(r'$\nu$', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# right panel: lambda_t (log scale)
plt.subplot(1, 2, 2)
# time index: 1961Q1 to 2019Q4
tq = pd.date_range('1961-01-01', periods=T, freq='QS')
shaded_band(tq, lam_lo, lam_hi, 0.85)
plt.plot(tq, lam_mean, 'k', linewidth=2)
plt.yscale('log')   # log scale for lambda
plt.xlim(pd.Timestamp('1960-01-01'), pd.Timestamp('2020-01-01'))
plt.ylabel(r'$\lambda_t$ (log scale)', fontsize=14)
plt.tight_layout()
plt.show()
