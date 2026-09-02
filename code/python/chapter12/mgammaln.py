# mgammaln.py
# Evaluates the log of the multivariate gamma function, log Gamma_n(x).
#
# Inputs:
#   n : dimension
#   x : argument
#
# Output:
#   k : log of the multivariate gamma function evaluated at x

import numpy as np
from scipy.special import gammaln


def mgammaln(n, x):
    return n*(n-1)/4*np.log(np.pi) + np.sum(gammaln(x - 0.5*np.arange(n)))
