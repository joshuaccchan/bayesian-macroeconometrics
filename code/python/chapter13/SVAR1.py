"""SVAR1.py
Updates the log-volatility h in the stationary AR(1)
stochastic volatility model
    y*_t      = h_t + log(chi^2_1),
    h_t - mu  = phi*(h_{t-1} - mu) + u_t,   u_t ~ N(0, sigh2),  |phi| < 1,
    h_1 - mu  ~ N(0, sigh2/(1 - phi^2)),
using the 7-component Gaussian mixture approximation of Kim, Shephard
and Chib (1998) and the precision-based sampler of Chan and
Jeliazkov (2009).

Inputs:
  ystar:  length-T vector of transformed observations, y*_t = log(y_t^2 + c)
  h:      length-T current draw of log-volatility
  mu:     scalar; unconditional mean of h
  phi:    scalar; AR(1) persistence, |phi| < 1
  sigh2:  scalar; innovation variance

Output:
  h:      length-T updated draw of log-volatility
"""

import numpy as np
from scipy.linalg import cholesky_banded, solveh_banded, solve_banded
from scipy.stats import norm


def SVAR1(ystar, h, mu, phi, sigh2):
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

    # sample h with precision K_h = H'*Sigma_inv*H/sigh2 + diag(iSig_s),
    # where H = I - phi*S1 and Sigma_inv = diag(1-phi^2, 1, ..., 1) encodes
    # the stationary initial variance sigh2/(1-phi^2); the prior is then
    # h ~ N(mu*1_T, sigh2*(H'*Sigma_inv*H)^{-1}).
    # HiSH is tridiagonal, so K_h is held in banded storage throughout:
    # main diagonal [1, 1+phi^2, ..., 1+phi^2, 1], off-diagonal -phi
    dm = np.concatenate(([1.0], (1 + phi**2)*np.ones(T-2), [1.0]))
    od = -phi*np.ones(T-1)
    ab = np.zeros((2, T))              # upper banded storage, bandwidth 1
    ab[1, :] = dm/sigh2 + iSig_s       # main diagonal of K_h
    ab[0, 1:] = od/sigh2               # superdiagonal of K_h
    HiSH_ones = dm.copy()              # HiSH @ 1_T via its diagonals
    HiSH_ones[:-1] += od
    HiSH_ones[1:] += od
    b = mu/sigh2*HiSH_ones + iSig_s*(ystar - d_s)
    h_hat = solveh_banded(ab, b)
    U = cholesky_banded(ab)            # K_h = U'U with U upper banded
    h = h_hat + solve_banded((0, 1), U, np.random.randn(T))
    return h
