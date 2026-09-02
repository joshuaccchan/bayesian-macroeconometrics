"""fit_BayesRidge.py
Returns the posterior-mean ridge estimator under an isotropic
Gaussian prior beta_j ~ N(0, 1/lam) and known unit error
variance. The estimator is beta_hat = (lam*I + X'X)^{-1} X'y.

Inputs:
  y  : length-T response
  X  : T-by-k design matrix
  lam: ridge regularisation parameter (prior precision scale)
       (called lambda in the MATLAB version; lambda is reserved in Python)

Output:
  beta_hat: length-k ridge posterior mean
"""

import numpy as np


def fit_BayesRidge(y, X, lam):
    k = X.shape[1]
    XX = X.T @ X
    Xy = X.T @ y
    Dbeta = np.linalg.solve(lam*np.eye(k) + XX, np.eye(k))
    beta_hat = Dbeta @ Xy
    return beta_hat
