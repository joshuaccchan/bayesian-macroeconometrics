# loglike_MA1.py
# Gaussian log-likelihood of the linear regression model with MA(1)
# errors, evaluated on regression residuals e = y - X*beta. The MA(1)
# structure is enforced via the band matrix H_psi, and the whitened
# residuals u = H_psi^{-1} e are obtained by a banded back-solve.
#
# Inputs:
#   psi : scalar MA(1) parameter
#   e   : length-T vector of regression residuals (y - X*beta)
#   sig2: innovation variance
#
# Outputs:
#   loglik: log p(e | psi, sig2) evaluated using u = H_psi^{-1} e
#   u     : implied innovations u

import numpy as np
from scipy.linalg import solve_banded


def loglike_MA1(psi, e, sig2):
    T = len(e)
    # H_psi is lower bidiagonal (ones on the diagonal, psi on the first
    # subdiagonal); solve H_psi u = e with a banded solver
    ab = np.zeros((2, T))
    ab[0, :] = 1.0    # diagonal
    ab[1, :] = psi    # subdiagonal (last entry unused)
    u = solve_banded((1, 0), ab, e)
    loglik = -0.5*T*np.log(2*np.pi*sig2) - 0.5*(u @ u)/sig2
    return loglik, u
