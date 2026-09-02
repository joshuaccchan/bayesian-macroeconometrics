# SVRW.py
# One MCMC update of the log-volatility vector h in the random-walk stochastic
# volatility model
#   y*_t = h_t + log(chi^2_1),
#   h_t  = h_{t-1} + u_t,   u_t ~ N(0, sigh2),   h_1 = h0 + u_1,
# using the 7-component Gaussian mixture approximation of Kim, Shephard and
# Chib (1998) and the precision-based sampler of Chan and Jeliazkov (2009).
# The mixture indicators are drawn first, then h is drawn from its Gaussian full
# conditional with a banded precision matrix.
#
# Inputs:
#   ystar : length-T vector of transformed observations, y*_t = log(y_t^2 + c)
#   h     : length-T current draw of the log-volatility
#   h0    : initial log-volatility (state at time 0), scalar
#   sigh2 : innovation variance of the random-walk state equation, scalar
#
# Output:
#   h     : length-T updated draw of the log-volatility

import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky_banded, solveh_banded, solve_banded


def SVRW(ystar, h, h0, sigh2):
    T = h.size

    # 7-component normal mixture approximation for log(chi^2_1)
    pj = np.array([0.0073, .10556, .00002, .04395, .34001, .24566, .2575])
    mj = np.array([-10.12999, -3.97281, -8.56686, 2.77786, .61942,
                   1.79518, -1.08819]) - 1.2704
    sigj2 = np.array([5.79596, 2.61369, 5.17950, .16735, .64009, .34023,
                      1.26261])
    sigj = np.sqrt(sigj2)

    # sample mixture indicators s_t in {0,...,6} (0-based)
    tmprand = np.random.rand(T)
    q = pj*norm.pdf(ystar[:, None], h[:, None] + mj, sigj)
    q = q/q.sum(axis=1, keepdims=True)
    cdfq = np.cumsum(q, axis=1)
    s = np.sum(tmprand[:, None] > cdfq, axis=1)
    d_s = mj[s]
    iSig_s = 1/sigj2[s]                 # diagonal of Sigma_s^{-1}

    # sample h: Kh = H'H/sigh2 + Sigma_s^{-1} is tridiagonal, so build its
    # banded storage directly (H'H has diagonal [2,...,2,1], off-diagonals -1,
    # and H'H*1_T = e_1)
    ab = np.zeros((2, T))               # upper banded storage of Kh
    ab[1, :] = 2/sigh2 + iSig_s
    ab[1, T-1] = 1/sigh2 + iSig_s[T-1]
    ab[0, 1:] = -1/sigh2
    bh = iSig_s*(ystar - d_s)
    bh[0] = bh[0] + h0/sigh2
    h_hat = solveh_banded(ab, bh)
    CKh = cholesky_banded(ab)           # Kh = CKh'*CKh with CKh upper banded
    h = h_hat + solve_banded((0, 1), CKh, np.random.randn(T))
    return h
