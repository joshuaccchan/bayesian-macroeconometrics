# linreg_ma1_dic.py
# Computes the DIC for the AR(2) model with MA(1) errors (ARMA(2,1))
# for US PCE inflation. The deviance is stored at each post-burn-in
# Metropolis-within-Gibbs iteration; the DIC equals the posterior
# mean deviance plus the effective number of parameters.

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
from loglike_MA1 import loglike_MA1
from sample_psi_RW import sample_psi_RW

np.random.seed(42)
nsim = 50000
burnin = 1000

# load data (MATLAB Range B2:B241 = first 240 rows: 1960Q1-2019Q4)
data = pd.read_csv('USPCE.csv').iloc[:240, 1].to_numpy()
y0 = data[:4]   # initial conditions
y = data[4:]    # sample used for estimation
T = y.size

# regressors for AR(2): [1, y_{t-1}, y_{t-2}]
xlag1 = np.concatenate(([y0[3]], y[:-1]))
xlag2 = np.concatenate((y0[2:4], y[:-2]))
X = np.column_stack((np.ones(T), xlag1, xlag2))
k = X.shape[1]  # number of regressors

# prior hyperparameters (indep normal and inverse-gamma)
beta0 = np.zeros(k)
iVbeta = 1/100*np.eye(k)
nu0 = 4
S0 = 1

# initialize the Markov chain
psi = 0
beta = np.linalg.solve(X.T @ X, X.T @ y)
sig2 = np.sum((y - X @ beta)**2)/T

# H_psi is lower bidiagonal (1 on the diagonal, psi on the first
# subdiagonal); store it in banded form for solve_banded
Hpsi = np.vstack((np.ones(T), np.full(T, psi)))
store_theta = np.zeros((nsim, k+2))  # [beta', sig2, psi]
store_dev = np.zeros(nsim)
count_psi = 0
g_var = 0.02  # proposal variance for RW-MH step for psi

for isim in range(nsim + burnin):
    # sample beta
    X_tilde = solve_banded((1, 0), Hpsi, X)
    y_tilde = solve_banded((1, 0), Hpsi, y)
    Dbeta = np.linalg.inv(iVbeta + X_tilde.T @ X_tilde/sig2)
    beta_hat = Dbeta @ (iVbeta @ beta0
        + X_tilde.T @ y_tilde/sig2)
    beta = beta_hat + np.linalg.cholesky(Dbeta) @ np.random.randn(k)

    # sample sig2
    e = y - X @ beta
    u = solve_banded((1, 0), Hpsi, e)
    sig2 = 1/np.random.gamma(nu0 + T/2, 1/(S0 + u @ u/2))

    # sample psi
    psi, accept = sample_psi_RW(psi, e, sig2, g_var)
    Hpsi = np.vstack((np.ones(T), np.full(T, psi)))

    # store the parameters
    if isim >= burnin:
        isave = isim - burnin
        count_psi = count_psi + accept  # post-burnin
        store_theta[isave, :] = np.concatenate((beta, [sig2, psi]))
        store_dev[isave] = -2*loglike_MA1(psi, y - X @ beta, sig2)

# acceptance rate
accept_rate = count_psi/nsim
print(f'Acceptance rate for psi: {accept_rate:.2f}')

theta_hat = store_theta.mean(axis=0)
beta_hat = theta_hat[:3]
sig2_hat = theta_hat[3]
psi_hat = theta_hat[4]
pD = store_dev.mean() + 2*loglike_MA1(psi_hat, y - X @ beta_hat, sig2_hat)
DIC = store_dev.mean() + pD
print(f'DIC: {DIC:.2f}')
print(f'pD: {pD:.1f}')
