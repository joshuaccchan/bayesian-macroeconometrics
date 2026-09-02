# sample_SVM_h_ARMH.py
# One MCMC update of the log-volatility vector h in the stochastic volatility
# in mean model, using a Laplace-based acceptance-rejection Metropolis-Hastings
# step. A Gaussian proposal N(h_hat, Kh^{-1}) is built from a second-order
# Taylor expansion of the log conditional posterior about its mode (located by
# Newton-Raphson); candidates are drawn by acceptance-rejection screening and
# then corrected by an independence-chain MH step.
#
# Inputs:
#   y     : length-T vector of observations
#   alp   : volatility-in-mean coefficient (scalar or length-T vector)
#   mu    : conditional-mean component x_t'beta (scalar or length-T vector)
#   h     : length-T current draw of the log-volatility
#   h0    : initial log-volatility (state at time 0), scalar
#   sigh2 : innovation variance of the random-walk state equation, scalar
#   HH    : T-by-T sparse prior precision component H'*H of the random-walk prior
#
# Outputs:
#   h      : length-T updated draw of the log-volatility
#   accept : 1 if the final MH step accepts the proposal, 0 otherwise

import numpy as np
from scipy.linalg import cholesky_banded, solveh_banded, solve_banded


def sample_SVM_h_ARMH(y, alp, mu, h, h0, sigh2, HH):
    T = h.size
    accept = 0
    kappa = 3

    # HH is tridiagonal; keep its diagonals for the banded storage of Kh
    HH_diag = HH.diagonal()
    HH_super = HH.diagonal(1)

    # step 1: locate posterior mode by Newton-Raphson
    s2 = (y - mu)**2
    ht = h.copy()   # initial value for Newton-Raphson
    max_norm = np.inf
    tol = 1e-4
    max_iter = 100   # safeguard: log-posterior is concave, so
    it = 0           # Newton-Raphson converges in a few steps
    ab = np.zeros((2, T))   # upper banded storage of Kh
    while max_norm > tol and it < max_iter:
        it = it + 1
        exp_ht = np.exp(ht)
        curv1 = 0.5*alp**2*exp_ht
        curv2 = 0.5*s2/exp_ht

        g = -0.5 - curv1 + curv2
        grad = g - (HH @ (ht - h0))/sigh2   # gradient

        # Kh = HH/sigh2 + diag(curv1 + curv2): negative Hessian, tridiagonal
        ab[1, :] = HH_diag/sigh2 + curv1 + curv2
        ab[0, 1:] = HH_super/sigh2

        new_ht = ht + solveh_banded(ab, grad)
        max_norm = np.max(np.abs(new_ht - ht))
        ht = new_ht
    h_hat = ht

    # step 2: construct Gaussian approximation
    # N(h_hat, Kh^{-1})
    Ch = cholesky_banded(ab)   # Kh = Ch'*Ch with Ch upper banded
    logdetKh = 2*np.sum(np.log(Ch[1, :]))   # row 1 of Ch holds diag(Ch)

    # log posterior kernel (normalizing constant omitted)
    def logp(x):
        return (-0.5*(x - h0) @ (HH @ (x - h0))/sigh2 - 0.5*np.sum(x)
                - 0.5*np.sum(np.exp(-x)*(y - mu - alp*np.exp(x))**2))

    # log Gaussian proposal density g(h); the quadratic form uses the
    # banded storage of Kh
    def logg(x):
        d = x - h_hat
        quad = np.sum(ab[1, :]*d**2) + 2*np.sum(ab[0, 1:]*d[:-1]*d[1:])
        return -0.5*T*np.log(2*np.pi) + 0.5*logdetKh - 0.5*quad

    # step 3: choose c = kappa * p(h_hat)/g(h_hat)
    logc = np.log(kappa) + logp(h_hat) - logg(h_hat)

    # proposal kernel: q(h) \propto  min{p(h)/(c g(h)), 1} g(h)
    def logq(x):
        return min(logp(x) - logc - logg(x), 0) + logg(x)

    # step 4: acceptance-rejection screening from g
    accepted_AR = False
    while not accepted_AR:
        hc = h_hat + solve_banded((0, 1), Ch, np.random.randn(T))  # draw from g
        log_acc_AR = logp(hc) - logc - logg(hc)
        if np.log(np.random.rand()) < min(log_acc_AR, 0):
            accepted_AR = True

    # step 5: MH correction
    log_alpha_ARMH = logp(hc) - logp(h) + logq(h) - logq(hc)
    if np.log(np.random.rand()) < log_alpha_ARMH:
        h = hc
        accept = 1
    return h, accept
