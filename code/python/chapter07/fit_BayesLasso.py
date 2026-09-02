"""fit_BayesLasso.py
Two-block Gibbs sampler for the Bayesian Lasso. The prior is the
scale-mixture-of-normals representation of the Laplace prior:
  (beta_j | sig2, tau_j^2) ~ N(0, sig2 * tau_j^2),
  (tau_j^2 | lam^2)        ~ G(1, lam^2/2),
  sig2 ~ IG(nu0, S0).
In each iteration, (beta, sig2) are sampled jointly from their NIG
full conditional, and each tau_j^2 is updated via the reciprocal
of an inverse-Gaussian draw. Requires igaussrnd.py.

Inputs:
  y      : length-T response
  X      : T-by-k design matrix
  lam    : Lasso shrinkage hyperparameter (lambda in the MATLAB version)
  S0,nu0 : inverse-gamma hyperparameters for sig2
  nsim   : number of post burn-in draws
  burnin : number of burn-in iterations

Outputs:
  beta_mean : length-k posterior mean of beta
  store_beta: nsim-by-k matrix of post burn-in beta draws
"""

import numpy as np
from igaussrnd import igaussrnd


def fit_BayesLasso(y, X, lam, S0, nu0, nsim, burnin):
    T, k = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    yy = y @ y
    store_beta = np.zeros((nsim, k))

    # initialize
    # np.random.gamma uses (shape, scale), like MATLAB's gamrnd
    tau2 = np.random.gamma(1, 2/lam**2, k)
    for isim in range(nsim + burnin):
        # sample beta and sigma^2
        Dbeta = np.linalg.solve(np.diag(1/tau2) + XtX, np.eye(k))
        beta_hat = Dbeta @ Xty
        S_hat = S0 + (yy - beta_hat @ np.linalg.solve(Dbeta, beta_hat))/2
        sig2 = 1/np.random.gamma(nu0 + T/2, 1/S_hat)
        beta = beta_hat + np.linalg.cholesky(sig2*Dbeta) @ np.random.randn(k)
        # sample tau2
        tmp = lam*np.sqrt(sig2)/np.abs(beta)
        tau2 = 1/igaussrnd(lam**2*np.ones(k), tmp)
        # store draws
        if isim >= burnin:
            isave = isim - burnin
            store_beta[isave, :] = beta
    beta_mean = store_beta.mean(axis=0)
    return beta_mean, store_beta
