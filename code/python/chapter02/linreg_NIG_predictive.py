# linreg_NIG_predictive.py
# Closed-form Bayesian inference for an AR(2) model of US PCE inflation
# under the natural conjugate (normal-inverse-gamma) prior; computes the
# one-step-ahead Student-t posterior predictive density for 2020Q1 and
# reports the predictive percentile at the realized value.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

from tpdfLS import tpdfLS

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

# prior hyperparameters (normal-inverse-gamma)
beta0 = np.zeros(k)       # prior mean
iVbeta = np.eye(k)/100    # prior precision
nu0 = 4                   # prior degrees of freedom
S0 = 1                    # prior scale

# Posterior hyperparameters
Dbeta = np.linalg.solve(iVbeta + X.T @ X, np.eye(k))
beta_hat = Dbeta @ (iVbeta @ beta0 + X.T @ y)
S_hat = S0 + (y @ y + beta0 @ iVbeta @ beta0
              - beta_hat @ np.linalg.solve(Dbeta, beta_hat))/2
nu_hat = nu0 + T/2

# Posterior predictive density for y_{T+1}
xTp1 = np.array([1, y[-1], y[-2]])   # regressor vector at T+1
ygrid = np.linspace(-3, 8, 500)      # evaluation grid
fy = tpdfLS(ygrid, xTp1 @ beta_hat,
            S_hat/nu_hat*(1 + xTp1 @ Dbeta @ xTp1), 2*nu_hat)

# Plot predictive density
plt.figure()
plt.plot(ygrid, fy, 'k', linewidth=2)
plt.axvline(1.39, color='k', linestyle='--', linewidth=2)
plt.xlabel(r'$y_{T+1}$', fontsize=14)
plt.ylabel(r'$p(y_{T+1}\mid\mathbf{y},\mathbf{x}_{T+1})$', fontsize=14)

y_obs = 1.39
z = (y_obs - xTp1 @ beta_hat) / np.sqrt(S_hat/nu_hat*(1 + xTp1 @ Dbeta @ xTp1))
pct = t.cdf(z, 2*nu_hat)
print('Posterior mean of beta:', beta_hat)
print(f'Predictive percentile at y_{{T+1}} = 1.39: {100*pct:.2f}%')

plt.show()
