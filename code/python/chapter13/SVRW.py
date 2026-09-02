"""SVRW.py
One MCMC update of the log-volatility vector h in the random-walk stochastic
volatility model
  y*_t = h_t + log(chi^2_1),
  h_t  = h_{t-1} + u_t,   u_t ~ N(0, sigh2),   h_1 = h0 + u_1,
using the 7-component Gaussian mixture approximation of Kim, Shephard and
Chib (1998) and the precision-based sampler of Chan and Jeliazkov (2009). The
mixture indicators are drawn first, then h is drawn from its Gaussian full
conditional with a banded precision matrix. The initial state h0 is treated
as known.

Inputs:
  ystar : length-T vector of transformed observations, y*_t = log(y_t^2 + c)
  h     : length-T current draw of the log-volatility
  h0    : initial log-volatility (state at time 0), scalar
  sigh2 : innovation variance of the random-walk state equation, scalar

Output:
  h     : length-T updated draw of the log-volatility
"""

import numpy as np
from scipy.linalg import cholesky_banded, solveh_banded, solve_banded
from scipy.stats import norm


def SVRW(ystar, h, h0, sigh2):
    T = len(h)

    # 7-component normal mixture approximation for log(chi^2_1)
    pj = np.array([0.0073, .10556, .00002, .04395, .34001, .24566, .2575])
    mj = np.array([-10.12999, -3.97281, -8.56686, 2.77786, .61942,
                   1.79518, -1.08819]) - 1.2704
    sigj2 = np.array([5.79596, 2.61369, 5.17950, .16735, .64009,
                      .34023, 1.26261])
    sigj = np.sqrt(sigj2)

    # sample mixture indicators s_t in {0,...,6}
    tmprand = np.random.rand(T)
    q = pj * norm.pdf(ystar[:, None], h[:, None] + mj, sigj)
    q = q / q.sum(axis=1, keepdims=True)
    cdfq = np.cumsum(q, axis=1)
    s = (tmprand[:, None] > cdfq).sum(axis=1)
    d_s = mj[s]
    iSig_s = 1/sigj2[s]

    # sample h with precision K_h = H'H/sigh2 + diag(iSig_s), where (Hh)_1 = h_1
    # and (Hh)_t = h_t - h_{t-1}. H'H is tridiagonal with main diagonal
    # [2, ..., 2, 1] and off-diagonal -1, so K_h is held in banded storage
    # throughout; H'H*1_T = e_1 contributes h0/sigh2 to the first entry of
    # the linear term
    dm = np.concatenate((2*np.ones(T-1), [1.0]))
    od = -np.ones(T-1)
    ab = np.zeros((2, T))              # upper banded storage, bandwidth 1
    ab[1, :] = dm/sigh2 + iSig_s       # main diagonal of K_h
    ab[0, 1:] = od/sigh2               # superdiagonal of K_h
    b = iSig_s*(ystar - d_s)
    b[0] += h0/sigh2
    h_hat = solveh_banded(ab, b)
    U = cholesky_banded(ab)            # K_h = U'U with U upper banded
    h = h_hat + solve_banded((0, 1), U, np.random.randn(T))
    return h
