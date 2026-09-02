"""SV_RW_gaussian_approx.py
Posterior mean of the log-volatility vector h under a single-Gaussian
(moment-matching) approximation to log(chi^2_1) and a random-walk prior for
h_t. The transformed observation y*_t = log(y_t^2 + c) is treated as
N(h_t - 1.2704, 4.9348), giving a linear Gaussian state space model whose
posterior mean is obtained by solving a banded linear system.

Inputs:
  s2    : length-T vector of squared observations y_t^2
  h0    : initial log-volatility (state at time 0), scalar
  sigh2 : innovation variance of the random-walk state equation, scalar

Output:
  h_hat : length-T posterior mean of the log-volatility
"""

import numpy as np
from scipy.linalg import solveh_banded


def SV_RW_gaussian_approx(s2, h0, sigh2):
    c = 1e-4   # avoids log(0)
    mu_eps = -1.2704  # E[log chi^2_1]
    var_eps = 4.9348  # Var[log chi^2_1]
    T = len(s2)

    ystar = np.log(s2 + c)
    # prior precision P = H'H/sigh2 with H = I - S1 is tridiagonal:
    # main diagonal [2, ..., 2, 1]/sigh2, off-diagonal -1/sigh2;
    # K_h = P + I/var_eps is held in banded storage
    ab = np.zeros((2, T))
    ab[1, :] = np.concatenate((2*np.ones(T-1), [1.0]))/sigh2 + 1/var_eps
    ab[0, 1:] = -1/sigh2
    # P @ b with prior mean b = h0*1_T: H'H maps 1_T to [1, 0, ..., 0]',
    # so P @ b has first element h0/sigh2 and zeros elsewhere
    Pb = np.zeros(T)
    Pb[0] = h0/sigh2
    h_hat = solveh_banded(ab, Pb + (ystar - mu_eps)/var_eps)
    return h_hat
