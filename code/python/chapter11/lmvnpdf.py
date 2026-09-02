# lmvnpdf.py
# Evaluates the log density of a multivariate normal distribution
#
# Inputs:
# x: evaluation points, k-vector
# mu: mean vector, k-vector
# Sig: covariance matrix, k x k
#
# Output:
# logden: log density of N(mu,Sig) evaluated at x

import numpy as np
from scipy.linalg import solve_triangular


def lmvnpdf(x, mu, Sig):
    x = np.asarray(x).flatten()
    mu = np.asarray(mu).flatten()
    k = mu.size

    CSig = np.linalg.cholesky(Sig)   # lower triangular
    e = solve_triangular(CSig, x - mu, lower=True)

    logden = -0.5*k*np.log(2*np.pi) - np.sum(np.log(np.diag(CSig))) - 0.5*(e @ e)
    return logden
