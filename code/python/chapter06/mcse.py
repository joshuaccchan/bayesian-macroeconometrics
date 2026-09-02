# mcse.py
# Computes Monte Carlo standard errors (MCSEs) for posterior means
# obtained from MCMC output. For each column of draws, MCSE(j) is
# sqrt(Omega_j / R), where Omega_j is the long-run variance estimated
# by the spectral variance at zero. Requires specvar0.py.

import numpy as np
from specvar0 import specvar0


def mcse(draws, L):
    """Monte Carlo standard errors for each column of draws.

    Inputs:
      draws : R-by-k matrix of MCMC draws
      L     : truncation lag for the spectral variance estimator

    Output:
      MCSE  : k-vector of Monte Carlo standard errors
    """
    R, k = draws.shape
    MCSE = np.zeros(k)
    for j in range(k):
        x = draws[:, j]
        Omega = specvar0(x, L)*R  # long-run variance of x^{(r)}
        MCSE[j] = np.sqrt(Omega/R)
    return MCSE
