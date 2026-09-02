# specvar0.py
# Estimates the long-run variance of the sample mean of x using a
# Bartlett-window spectral-variance estimator at frequency zero:
#       S = (gamma_0 + 2 * sum_{ell=1}^{L} w_ell * gamma_ell) / T,
# where gamma_ell is the lag-ell sample autocovariance and the weights
# w_ell = 1 - ell/(L+1) are Bartlett (Newey-West) weights.

import numpy as np


def specvar0(x, L):
    """Long-run variance of the sample mean of x.

    Inputs:
      x : T-vector (demeaned internally)
      L : truncation lag (nonnegative integer)

    Output:
      S : long-run variance of the sample mean of x
    """
    x = np.asarray(x).flatten()
    T = len(x)
    x = x - x.mean()

    # autocovariances up to lag L
    gamma0 = (x @ x)/T
    Sx = gamma0

    for ell in range(1, L + 1):
        w = 1 - ell/(L + 1)  # Bartlett weight
        gamma = (x[ell:] @ x[:-ell])/T
        Sx = Sx + 2*w*gamma

    # long-run variance of the mean
    S = Sx/T
    return S
