# loglike_MA1.py
# Gaussian log-likelihood of the linear regression model with MA(1)
# errors, evaluated on regression residuals e = y - X*beta. The MA(1)
# structure is enforced via the band matrix H_psi, and the whitened
# residuals u = H_psi^{-1} e are obtained by a banded back-solve.

import numpy as np
from scipy.linalg import solve_banded


def loglike_MA1(psi, e, sig2):
    """Log-likelihood log p(e | psi, sig2) of the MA(1)-error model.

    Inputs:
      psi : scalar MA(1) parameter
      e   : T-vector of regression residuals (y - X*beta)
      sig2: innovation variance

    Output:
      loglik: log p(e | psi, sig2) evaluated using u = H_psi^{-1} e
              (the MATLAB version also returns the innovations u)
    """
    T = len(e)
    # H_psi is lower bidiagonal (1 on the diagonal, psi on the first
    # subdiagonal); store it in banded form for solve_banded
    Hpsi = np.vstack((np.ones(T), np.full(T, psi)))
    u = solve_banded((1, 0), Hpsi, e)
    loglik = -0.5*T*np.log(2*np.pi*sig2) - 0.5*(u @ u)/sig2
    return loglik
