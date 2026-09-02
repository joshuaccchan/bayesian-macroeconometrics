"""logpost_gam.py
Evaluates the log of the collapsed posterior kernel p(gam | y, sig2)
for the SSVS regression, with the regression coefficients beta
integrated out analytically. The returned value sums the marginal
log-likelihood log p(y | gam, sig2) and the log-prior log p(gam)
under independent Bernoulli(bp) inclusion priors. If the implied
precision matrix is not positive definite, returns -inf.

Inputs:
  gam   : length-k vector of inclusion indicators in {0,1}
  y     : length-T response
  X     : T-by-k design matrix
  sig2  : scalar error variance
  bp    : length-k vector of prior inclusion probabilities
  iVbeta: k-by-k prior precision of beta

Output:
  logden: log p(gam | y, sig2) up to a normalising constant
"""

import numpy as np
from scipy.linalg import cholesky


def logpost_gam(gam, y, X, sig2, bp, iVbeta):
    Xtilde = X * gam  # X @ diag(gam): zero out excluded columns

    iDbeta = iVbeta + (Xtilde.T @ Xtilde)/sig2
    try:
        C = cholesky(iDbeta)  # upper triangular, like MATLAB's chol
    except np.linalg.LinAlgError:
        return -np.inf

    beta_hat = np.linalg.solve(iDbeta, Xtilde.T @ y/sig2)
    logden = (-np.sum(np.log(np.diag(C))) + 0.5*beta_hat @ iDbeta @ beta_hat
              + np.sum(gam[1:]*np.log(bp[1:]) + (1-gam[1:])*np.log(1-bp[1:])))
    return logden
