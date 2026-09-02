# linreg_bqr.py
# Gibbs sampler for Bayesian quantile regression of growth-at-risk.
# The model is
#   y_{t+h} = beta_1 + beta_2 gdp_t + beta_3 nfci_t + eps_{t+h},
#   eps_{t+h} ~ AL(0, sig2, tau),
# with the location-scale mixture-of-normals representation
# (eps | lam) ~ N(vartheta*lam, varphi*sig2*lam), lam ~ G(1, 1/sig2),
# and independent N + IG priors on (beta, sig2). The latent scales
# {lam_t} are updated via the inverse Gaussian representation; see
# igaussrnd.py.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from igaussrnd import igaussrnd

np.random.seed(42)

nsim = 20000
burnin = 1000
tau = 0.05   # quantile level (e.g., 0.05, 0.10, 0.50)
h = 4        # forecast horizon in quarters (e.g., 1 or 4)

# AL mixture constants
vartheta = (1 - 2*tau) / (tau*(1 - tau))
varphi = 2 / (tau*(1 - tau))

# load GDP-NFCI merged data
#   Date is a quarter-start date
#   GDP is quarterly (annualized) real GDP growth
#   NFCI is quarterly average of weekly NFCI
TBL = pd.read_csv('GDP_NFCI_merged.csv')
TBL['Date'] = pd.to_datetime(TBL['Date'], format='%m/%d/%Y')

# keep observations with NFCI available
idx = TBL['NFCI'].notna() & TBL['GDP'].notna()
TBL = TBL[idx].reset_index(drop=True)
gdp = TBL['GDP'].to_numpy()
nfci = TBL['NFCI'].to_numpy()

# Construct regression for Growth-at-Risk:
#   y_{t+h} = beta0 + beta1*gdp_t + beta2*nfci_t + eps_{t+h}
T0 = len(gdp)
T = T0 - h
y = gdp[h:]   # y_{t+h}
X = np.column_stack((np.ones(T), gdp[:-h], nfci[:-h]))
k = X.shape[1]

# prior hyperparameters (indep normal and inverse-gamma)
beta0 = np.zeros(k)
iVbeta = np.eye(k)/100   # prior precision
nu0 = 3
S0 = 1*(nu0 - 1)

# initialize the Markov chain
beta = np.linalg.solve(X.T @ X, X.T @ y)
sig2 = np.mean((y - X @ beta)**2)
lam = np.random.gamma(1, sig2, T)
ilam = 1/lam   # iLam is diagonal; keep 1/lam as a vector
store_theta = np.zeros((nsim, k+1))   # [beta' sig2]

# Gibbs sampler starts here
for isim in range(nsim + burnin):

    # sample beta
    ytilde = y - vartheta*lam
    Dbeta = np.linalg.solve(iVbeta + (X*ilam[:, None]).T @ X/(varphi*sig2),
                            np.eye(k))
    beta_hat = Dbeta @ (iVbeta @ beta0
                        + X.T @ (ilam*ytilde)/(varphi*sig2))
    C = np.linalg.cholesky(Dbeta)
    beta = beta_hat + C @ np.random.randn(k)

    # sample sig2
    e = y - X @ beta - vartheta*lam
    S_hat = S0 + np.sum(lam) + e @ (ilam*e)/(2*varphi)
    sig2 = 1/np.random.gamma(nu0 + 3*T/2, 1/S_hat)

    # sample lambda
    a_tau = (vartheta**2 + 2*varphi)/(varphi*sig2)
    mu = np.sqrt(vartheta**2 + 2*varphi) / np.abs(y - X @ beta)
    lam = 1/igaussrnd(a_tau*np.ones(T), mu)
    ilam = 1/lam

    if isim >= burnin:
        isave = isim - burnin
        store_theta[isave, :] = np.append(beta, sig2)

theta_mean = store_theta.mean(axis=0)
theta_CI = np.quantile(store_theta, [.025, .975], axis=0)

print('Posterior mean (beta; sig2):')
print(theta_mean)
print('Posterior 95% CI (rows: 2.5%, 97.5%):')
print(theta_CI)

# compute GaR_t(h; tau)
B = store_theta[:, :k]
Xgar = X   # regressors at time t (aligned with y_{t+h})
GaR_draws = Xgar @ B.T   # dimension is T x nsim
GaR_mean = GaR_draws.mean(axis=1)

# Credible bands (pointwise)
GaR_lo95 = np.quantile(GaR_draws, 0.025, axis=1)
GaR_hi95 = np.quantile(GaR_draws, 0.975, axis=1)

# align dates with forecasted outcome y_{t+h}
date_f = TBL['Date'].iloc[h:].reset_index(drop=True)

plt.figure()
plt.plot(date_f, GaR_mean, 'k-', linewidth=2)
plt.ylabel('GaR', fontsize=14)

# Start two quarters before first forecast date
x_start = date_f.iloc[0] - pd.DateOffset(months=6)
x_end = date_f.iloc[-1]
plt.xlim(x_start, x_end)
plt.show()
