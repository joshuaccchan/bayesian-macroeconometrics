"""inefficiency_factor.py
Computes inefficiency factors (integrated autocorrelation times) for
MCMC output. For each column of draws, IF[j] = Omega_j / sigma2_j,
where sigma2_j is the marginal variance and Omega_j is the long-run
variance estimated by the spectral variance at zero (Bartlett window).
Requires specvar0.py.

Inputs:
  draws : R x k matrix of MCMC draws (each column is a scalar
          sequence); a 1-D array is treated as a single column
  L     : truncation lag for the spectral variance estimator

Output:
  IF    : length-k vector of inefficiency factors
"""

import numpy as np

from specvar0 import specvar0


def inefficiency_factor(draws, L):
    draws = np.asarray(draws)
    if draws.ndim == 1:
        draws = draws[:, None]
    R, k = draws.shape
    IF = np.zeros(k)
    for j in range(k):
        x = draws[:, j]
        sigma2 = np.var(x)           # marginal variance (population version)
        Omega = specvar0(x, L) * R   # long-run variance of x^{(r)}
        IF[j] = Omega / sigma2
    return IF
