# linreg_horseshoe.py
# Gibbs sampler for the regression of US PCE inflation on an
# intercept and the first two lags of 16 macroeconomic and financial
# indicators, with a horseshoe prior on the 32 slope coefficients
# and a diffuse normal prior on the intercept. The half-Cauchy local
# and global scales tau_j and theta are represented via the latent
# inverse-gamma scale mixture of Makalic and Schmidt (2016), yielding
# inverse-gamma full conditionals for (tau_j^2, theta^2, lam_tau,
# lam_theta). A small floor is imposed on tau_j^2 * theta^2 to avoid
# numerical ill-conditioning in the Gaussian update for beta.
# Requires PCE_regression_data.csv.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

nsim = 50000
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

# prior hyperparameters
Vbeta0 = 100  # prior variance for the intercept
nu0 = 3
S0 = 1

store_beta = np.zeros((nsim, k))
XtX = X.T @ X
Xty = X.T @ y

# initialize
# np.random.gamma uses (shape, scale), like MATLAB's gamrnd
beta_ols = np.linalg.solve(X.T @ X, X.T @ y)
sig2_ols = np.sum((y - X @ beta_ols)**2)/T
beta = beta_ols
sig2 = sig2_ols
lam_tau = 1/np.random.gamma(1/2, 1, k-1)
lam_theta = 1/np.random.gamma(1/2, 1)
tau2 = 1/np.random.gamma(1/2, lam_tau)
theta2 = 1/np.random.gamma(1/2, lam_theta)

for isim in range(nsim + burnin):
    # sample beta
    var_slope = tau2*theta2
    var_slope = np.maximum(var_slope, 1e-10)  # set lower bounds
    prior_prec = np.concatenate(([1/Vbeta0], 1/var_slope))
    Dbeta = np.linalg.solve(np.diag(prior_prec) + XtX/sig2, np.eye(k))
    beta_hat = Dbeta @ Xty/sig2
    beta = beta_hat + np.linalg.cholesky(Dbeta) @ np.random.randn(k)

    # sample sig2
    e = y - X @ beta
    S_hat = S0 + e @ e/2
    sig2 = 1/np.random.gamma(nu0 + T/2, 1/S_hat)

    # sample tau2
    tmp1 = 1/lam_tau + beta[1:]**2/(2*theta2)
    tau2 = 1/np.random.gamma(1, 1/tmp1)

    # sample theta2
    tmp2 = 1/lam_theta + np.sum(beta[1:]**2/tau2)/2
    theta2 = 1/np.random.gamma(k/2, 1/tmp2)

    # sample lam_tau, lam_theta
    lam_tau = 1/np.random.gamma(1, 1/(1 + 1/tau2))
    lam_theta = 1/np.random.gamma(1, 1/(1 + 1/theta2))

    # store the parameters
    if isim >= burnin:
        isave = isim - burnin
        store_beta[isave, :] = beta

beta_mean = store_beta[:, 1:].mean(axis=0)
beta_ci = np.quantile(store_beta[:, 1:], [.05, .95], axis=0).T

gray = (0.5, 0.5, 0.5)
short_names = ['PCE inflation', 'Oil price', 'FFR', '10y yield',
               'Term spread', 'AAA-FFR spread', 'Real M2', 'Consumer credit',
               'UM sentiment', 'Cap. utilization', 'Real GDP', 'Real PCE',
               'Ind. production', 'Unemployment', 'Payrolls', 'Housing starts']
ytick_labels = ([s + ' (lag 1)' for s in short_names]
                + [s + ' (lag 2)' for s in short_names])

# posterior summaries: the largest slope coefficients in absolute value
print('largest posterior-mean slope coefficients (with 90% credible intervals):')
for idx in np.argsort(-np.abs(beta_mean))[:5]:
    print(f'  {ytick_labels[idx]:<28s} {beta_mean[idx]: .3f}  '
          f'[{beta_ci[idx, 0]: .3f}, {beta_ci[idx, 1]: .3f}]')

fig1 = plt.figure(figsize=(6, 7))
ax = plt.gca()
# 0-line
ax.plot([0, 0], [0, k], '-', color=gray, linewidth=1)
# Credible intervals + means
for idx in range(1, k):
    ax.plot([beta_ci[idx-1, 0], beta_ci[idx-1, 1]], [idx, idx], '-',
            color=gray, linewidth=2)
    ax.plot(beta_mean[idx-1], idx, 'o',
            markerfacecolor=gray, markeredgecolor=gray)
ax.set_xlabel(r'$\beta_j$')
ax.set_ylim(0, k)
ax.set_yticks(range(1, k))
ax.set_yticklabels(ytick_labels, fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()
