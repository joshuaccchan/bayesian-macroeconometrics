# linreg_ssvs.py
# Collapsed Gibbs sampler for the point-mass spike-and-slab SSVS
# regression of US PCE inflation on an intercept and the first two
# lags of 16 macroeconomic and financial indicators.
# Requires logpost_gam.py and PCE_regression_data.csv.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logpost_gam import logpost_gam

np.random.seed(42)

nsim = 20000
burnin = 1000

# Load data (columns B:Q, rows 2-241 of the CSV = 1960Q1-2019Q4)
data = pd.read_csv('PCE_regression_data.csv').iloc[:240, 1:17].to_numpy()
y_raw = data[:, 0]
X_raw = data
p = 2  # of lags
T0, m = X_raw.shape
X = np.zeros((T0, m*p))  # construct predictors
for j in range(1, p+1):
    X[:, (j-1)*m:j*m] = np.vstack([np.full((j, m), np.nan), X_raw[:-j, :]])
# Drop initial rows with missing values from lagging
y = y_raw[p:]
X = X[p:, :]
T = len(y)
X = np.column_stack([np.ones(T), X])  # add an intercept
k = m*p + 1

# priors
iVbeta = np.eye(k)/100
nu0 = 3
S0 = 1
bp = 0.5*np.ones(k)  # prior inclusion probabilities

store_gam = np.zeros((nsim, k))
store_betatilde = np.zeros((nsim, k))

# initialize the Markov chain
beta_ols = np.linalg.solve(X.T @ X, X.T @ y)
sig2_ols = np.sum((y - X @ beta_ols)**2)/T
beta_ols_std = np.sqrt(sig2_ols*np.diag(np.linalg.solve(X.T @ X, np.eye(k))))
beta = beta_ols
sig2 = sig2_ols
gam = (np.abs(beta)/beta_ols_std > 1.65).astype(float)
gam[0] = 1  # fix gamma_1 = 1

for isim in range(nsim + burnin):
    # sample gamma marginal of beta (single-site)
    for j in range(1, k):
        # evaluate log-kernel at gamma_j = 0
        gam0 = gam.copy()
        gam0[j] = 0
        l0 = logpost_gam(gam0, y, X, sig2, bp, iVbeta)

        # evaluate log-kernel at gamma_j = 1
        gam1 = gam.copy()
        gam1[j] = 1
        l1 = logpost_gam(gam1, y, X, sig2, bp, iVbeta)

        # stable Bernoulli draw
        mm = max(l0, l1)
        p1 = np.exp(l1 - mm)/(np.exp(l0 - mm) + np.exp(l1 - mm))
        gam[j] = float(np.random.rand() < p1)

    # sample beta | gamma, sig2
    Xtilde = X * gam  # X @ diag(gam)
    Dbeta = np.linalg.solve(iVbeta + Xtilde.T @ Xtilde/sig2, np.eye(k))
    beta_hat = Dbeta @ (Xtilde.T @ y/sig2)
    C = np.linalg.cholesky(Dbeta)
    beta = beta_hat + C @ np.random.randn(k)

    # sample sig2
    e = y - Xtilde @ beta
    sig2 = 1/np.random.gamma(nu0 + T/2, 1/(S0 + e @ e/2))

    if isim >= burnin:
        # store the parameters
        isave = isim - burnin
        store_betatilde[isave, :] = beta*gam
        store_gam[isave, :] = gam

betatilde_mean = store_betatilde[:, 1:].mean(axis=0)
betatilde_ci = np.quantile(store_betatilde[:, 1:], [.05, .95], axis=0).T
gam_mean = store_gam.mean(axis=0)
# model size - exclude the intercept indicator
sumgam_mean = np.mean(np.sum(store_gam[:, 1:], axis=1))

gray = (0.5, 0.5, 0.5)
short_names = ['PCE inflation', 'Oil price', 'FFR', '10y yield',
               'Term spread', 'AAA-FFR spread', 'Real M2', 'Consumer credit',
               'UM sentiment', 'Cap. utilization', 'Real GDP', 'Real PCE',
               'Ind. production', 'Unemployment', 'Payrolls', 'Housing starts']
ytick_labels = ([s + ' (lag 1)' for s in short_names]
                + [s + ' (lag 2)' for s in short_names])

# posterior summaries
print(f'posterior mean model size (excl. intercept): {sumgam_mean:.2f}')
print('predictors with posterior inclusion probability > 0.5:')
for idx in np.flatnonzero(gam_mean[1:] > 0.5):
    print(f'  {ytick_labels[idx]:<28s} P(gam=1|y) = {gam_mean[idx+1]:.3f}  '
          f'E(betatilde|y) = {betatilde_mean[idx]: .3f}')

plt.figure(figsize=(6, 7))
ax = plt.gca()
# 0-line
ax.plot([0, 0], [0, k], '-', color=gray, linewidth=1)
# Credible intervals + means
for idx in range(1, k):
    ax.plot([betatilde_ci[idx-1, 0], betatilde_ci[idx-1, 1]], [idx, idx], '-',
            color=gray, linewidth=2)
    ax.plot(betatilde_mean[idx-1], idx, 'o',
            markerfacecolor=gray, markeredgecolor=gray)
ax.set_xlabel(r'$\tilde{\beta}_j$')
ax.set_ylim(0, k)
ax.set_yticks(range(1, k))
ax.set_yticklabels(ytick_labels, fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()
