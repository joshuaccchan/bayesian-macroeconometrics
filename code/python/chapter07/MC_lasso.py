# MC_lasso.py
# Monte Carlo benchmark for the Bayesian Lasso of Example 7.2:
# generates R = 100 datasets of size T = 100 with k regressors and
# true coefficient vector (1, 1, 0, ..., 0)', then compares the
# posterior-mean MSE of the Bayesian Lasso against ordinary least
# squares and ridge regression. The shrinkage parameter lambda is
# set as lambda = k * sqrt(sig2_LS) / sum |beta_LS|. Requires
# fit_BayesLasso.py, fit_BayesRidge.py, and igaussrnd.py.

import numpy as np
import matplotlib.pyplot as plt
from fit_BayesRidge import fit_BayesRidge
from fit_BayesLasso import fit_BayesLasso

np.random.seed(42)

nsim = 5000
burnin = 1000
R = 100
T = 100
k = 20  # number of regressors (change to 50 for the high-dim case)

# priors
nu0 = 5
S0 = 1

# true parameter values
truebeta = np.concatenate(([1, 1], np.zeros(k-2)))
truesig2 = 1

store_MSE = np.zeros((R, 3))
for idata in range(R):   # a plain loop; joblib could parallelize this
    # generate data
    X = 5*np.random.rand(T, k)
    y = X @ truebeta + np.sqrt(truesig2)*np.random.randn(T)

    beta_ols = np.linalg.solve(X.T @ X, X.T @ y)
    sig2_ols = np.sum((y - X @ beta_ols)**2)/T
    lam = k*np.sqrt(sig2_ols)/np.sum(np.abs(beta_ols))

    beta_ridge = fit_BayesRidge(y, X, 2*k*sig2_ols/(beta_ols @ beta_ols))
    beta_lasso, _ = fit_BayesLasso(y, X, lam, S0, nu0, nsim, burnin)

    store_MSE[idata, :] = [np.mean((truebeta - beta_ols)**2),
                           np.mean((truebeta - beta_ridge)**2),
                           np.mean((truebeta - beta_lasso)**2)]

MSE_med = np.median(store_MSE, axis=0)
print(f'\nMedian MSE  (k = {k}):')
print(f'  OLS    {MSE_med[0]:.4f}\n  Ridge  {MSE_med[1]:.4f}\n  Lasso  {MSE_med[2]:.4f}')

fig1 = plt.figure(figsize=(4, 6))
plt.boxplot(store_MSE, tick_labels=['LS', 'Ridge', 'Lasso'], whis=10,
            sym='k+', boxprops={'color': 'k'}, whiskerprops={'color': 'k'},
            capprops={'color': 'k'}, medianprops={'color': 'k'})
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()
