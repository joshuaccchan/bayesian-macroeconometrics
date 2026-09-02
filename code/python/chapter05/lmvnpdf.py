# lmvnpdf.py
# Evaluates the log density of a multivariate normal distribution.

import numpy as np
from scipy.linalg import solve_triangular


def lmvnpdf(x, mu, Sig):
    """Log density of N(mu, Sig) evaluated at x.

    Inputs:
      x  : evaluation point, k-vector
      mu : mean vector, k-vector
      Sig: covariance matrix, k x k

    Output:
      logden: log density of N(mu, Sig) evaluated at x
    """
    x = np.asarray(x).flatten()
    mu = np.asarray(mu).flatten()
    k = len(mu)

    CSig = np.linalg.cholesky(Sig)  # lower Cholesky factor
    e = solve_triangular(CSig, x - mu, lower=True)

    logden = -0.5*k*np.log(2*np.pi) - np.sum(np.log(np.diag(CSig))) \
        - 0.5*(e @ e)
    return logden
