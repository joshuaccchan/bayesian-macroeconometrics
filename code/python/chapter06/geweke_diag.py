# geweke_diag.py
# Computes Geweke's convergence diagnostic for MCMC output. For each
# column of draws, the chain is split into an early segment A (first
# fraction a) and a late segment B (last fraction 1-b), the segment
# means are compared, and the Z-statistic
#       Z = (mean_A - mean_B) / sqrt(S_A + S_B)
# is computed, where S_A and S_B are spectral-variance estimates of the
# long-run variances of the two segment means. Under stationarity, Z is
# approximately standard normal; large |Z| provides evidence against
# convergence for the chosen summary. Requires specvar0.py.

import numpy as np
from scipy.stats import norm
from specvar0 import specvar0


def geweke_diag(draws, a=0.10, b=0.50, Lrule='auto'):
    """Geweke's convergence diagnostic for each column of draws.

    Inputs:
      draws : R-by-k matrix of posterior draws (each column is a scalar
              summary computed from MCMC output)
      a     : fraction for early segment (default 0.10)
      b     : fraction defining late segment, B = floor(b*R):R
              (default 0.50)
      Lrule : truncation-lag rule for the spectral variance estimator,
              either 'auto' (default) or a nonnegative integer L

    Outputs:
      Z     : k-vector of Geweke Z-statistics
      pval  : k-vector of two-sided p-values (normal approximation)
      info  : dict with keys A, B, LA, LB, a, b giving the segment
              indices and chosen truncation lags
    """
    R, k = draws.shape
    if not (0 < a < b < 1):
        raise ValueError('Require 0 < a < b < 1.')

    A = np.arange(int(np.floor(a*R)))           # MATLAB: 1:floor(a*R)
    B = np.arange(int(np.floor(b*R)) - 1, R)    # MATLAB: floor(b*R):R

    nA = len(A)
    nB = len(B)

    # Choose truncation lags
    if isinstance(Lrule, str):
        if Lrule.lower() == 'auto':
            LA = max(0, int(np.floor(4*(nA/100)**(2/9))))
            LB = max(0, int(np.floor(4*(nB/100)**(2/9))))
        else:
            raise ValueError("Unknown Lrule. Use 'auto' or an integer.")
    else:
        LA = max(0, int(np.floor(Lrule)))
        LB = LA

    Z = np.full(k, np.nan)
    pval = np.full(k, np.nan)

    for j in range(k):
        xA = draws[A, j]
        xB = draws[B, j]

        mA = xA.mean()
        mB = xB.mean()

        SA = specvar0(xA, LA)  # long-run variance of mean for segment A
        SB = specvar0(xB, LB)  # long-run variance of mean for segment B

        Z[j] = (mA - mB)/np.sqrt(SA + SB)

        # Two-sided p-value under N(0,1)
        pval[j] = 2*(1 - norm.cdf(abs(Z[j])))

    info = {'A': A, 'B': B, 'LA': LA, 'LB': LB, 'a': a, 'b': b}
    return Z, pval, info
