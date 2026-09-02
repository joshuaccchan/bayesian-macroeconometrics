# SV_RW_gaussian_approx.py
# Posterior mean of the log-volatility vector h under a single-Gaussian
# (moment-matching) approximation to log(chi^2_1) and a random-walk prior for
# h_t. The transformed observation y*_t = log(y_t^2 + c) is treated as
# N(h_t - 1.2704, 4.9348), giving a linear Gaussian state space model whose
# posterior mean is obtained by solving a banded linear system.
#
# Inputs:
#   s2    : length-T vector of squared observations y_t^2
#   h0    : initial log-volatility (state at time 0), scalar
#   sigh2 : innovation variance of the random-walk state equation, scalar
#
# Output:
#   h_hat : length-T posterior mean of the log-volatility

import numpy as np
from scipy.linalg import solveh_banded


def SV_RW_gaussian_approx(s2, h0, sigh2):
    c = 1e-4           # avoids log(0)
    mu_eps = -1.2704   # E[log chi^2_1]
    var_eps = 4.9348   # Var[log chi^2_1]
    T = s2.size

    ystar = np.log(s2 + c)
    # Kh = H'H/sigh2 + I/var_eps is tridiagonal, so build its banded storage
    # directly (H'H has diagonal [2,...,2,1], off-diagonals -1, and the prior
    # precision times the prior mean h0*1_T is h0/sigh2*e_1)
    ab = np.zeros((2, T))               # upper banded storage of Kh
    ab[1, :] = 2/sigh2 + 1/var_eps
    ab[1, T-1] = 1/sigh2 + 1/var_eps
    ab[0, 1:] = -1/sigh2
    b = (ystar - mu_eps)/var_eps
    b[0] = b[0] + h0/sigh2
    h_hat = solveh_banded(ab, b)
    return h_hat
