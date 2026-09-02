# linreg_indep_predictive.py
# Gibbs sampler for an AR(2) model of US PCE inflation under the
# independent normal and inverse-gamma prior; constructs the one-step-ahead
# posterior predictive density for 2020Q1 by averaging conditional Gaussian
# forecasts across posterior draws.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

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

# prior hyperparameters (independent normal and inverse-gamma)
beta0 = np.zeros(k)
iVbeta = np.eye(k)/100   # prior precision of beta
nu0 = 4
S0 = 1

# one-step-ahead regressor vector (T+1)
xTp1 = np.array([1, y[-1], y[-2]])
y_obs = 1.39
ygrid = np.linspace(-3, 8, 300)

# initialize chain
beta = np.linalg.solve(X.T @ X, X.T @ y)
e = y - X @ beta
sig2 = (e @ e)/T

store_theta = np.zeros((nsim, k+1))   # [beta' sig2]
store_fy = np.zeros(ygrid.size)       # store only the sum

# Gibbs sampler
for isim in range(nsim + burnin):
    # sample beta
    Dbeta = np.linalg.solve(iVbeta + X.T @ X/sig2, np.eye(k))
    beta_hat = Dbeta @ (iVbeta @ beta0 + X.T @ y/sig2)
    C = np.linalg.cholesky(Dbeta)
    beta = beta_hat + C @ np.random.randn(k)

    # sample sig2
    # np.random.gamma uses (shape, scale)
    e = y - X @ beta
    sig2 = 1/np.random.gamma(nu0 + T/2, 1/(S0 + 0.5*(e @ e)))

    if isim >= burnin:
        isave = isim - burnin
        store_theta[isave, :] = np.append(beta, sig2)

        # store one-step-ahead predictive density
        mu1 = xTp1 @ beta
        s2_1 = sig2
        fy = norm.pdf(ygrid, mu1, np.sqrt(s2_1))
        store_fy = store_fy + fy

# posterior summaries
theta_mean = store_theta.mean(axis=0)
theta_CI = np.quantile(store_theta, [.025, .975], axis=0)

print("Posterior mean of [beta' sig2]:")
print(theta_mean)
print('95% posterior intervals (rows: 2.5%, 97.5%):')
print(theta_CI)

# Plot predictive density
fy_hat = store_fy/nsim
plt.figure()
plt.plot(ygrid, fy_hat, 'k', linewidth=2)
plt.axvline(y_obs, color='k', linestyle='--', linewidth=2)
plt.xlabel(r'$y_{T+1}$', fontsize=14)
plt.ylabel(r'$p(y_{T+1}\mid\mathbf{y},\mathbf{x}_{T+1})$', fontsize=14)
plt.show()
