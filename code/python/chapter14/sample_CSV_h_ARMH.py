"""sample_CSV_h_ARMH.py
Samples the common log-volatility h = (h_1,...,h_T)' in a VAR with a common
stochastic volatility error structure (Section 14.2) using a Laplace-based
acceptance-rejection Metropolis-Hastings (ARMH) step. The log-volatility
follows the stationary AR(1) h_t = phi*h_{t-1} + u_t^h, u_t^h ~ N(0,sigh2),
with h_1 ~ N(0, sigh2/(1-phi^2)). Given the per-period sums of squares s2
(summed over the n variables), the conditional density of h is non-standard.
A Gaussian approximation centered at the posterior mode serves as the
proposal: a candidate is first screened by acceptance-rejection and then
corrected by a Metropolis-Hastings step.

Inputs:
  s2:    length-T vector of per-period sums of squares
  phi:   AR(1) persistence of the log-volatility
  sigh2: innovation variance of the log-volatility
  h:     length-T current draw of the log-volatility
  n:     number of variables in the VAR
  kappa: (optional) envelope constant for the AR screening step; larger
         values improve mixing at the cost of more candidate draws (default 3)

Outputs:
  h:      length-T new draw of the log-volatility
  accept: 1 if the MH step accepts, 0 otherwise
"""

import numpy as np
from scipy.linalg import cholesky_banded, solveh_banded, solve_banded


def sample_CSV_h_ARMH(s2, phi, sigh2, h, n, kappa=3):
    T = len(h)
    accept = 0

    # AR(1) prior precision HiSH = Hphi' * diag(.) * Hphi (zero mean) is
    # tridiagonal; it is held via its diagonals throughout:
    # main diagonal [1, 1+phi^2, ..., 1+phi^2, 1]/sigh2, off-diagonal -phi/sigh2
    dm = np.concatenate(([1.0], (1 + phi**2)*np.ones(T-2), [1.0]))/sigh2
    od = -phi/sigh2*np.ones(T-1)

    def HiSH_mv(x):
        # tridiagonal matrix-vector product HiSH @ x
        y = dm*x
        y[:-1] += od*x[1:]
        y[1:] += od*x[:-1]
        return y

    # step 1: locate the posterior mode by Newton-Raphson
    ht = h.copy()
    max_norm = np.inf
    tol = 1e-4
    ab = np.zeros((2, T))   # banded storage for the negative Hessian K_h
    while max_norm > tol:
        eis2 = np.exp(-ht)*s2
        grad = -HiSH_mv(ht) - n/2 + 0.5*eis2       # gradient of log posterior
        ab[1, :] = dm + 0.5*eis2                   # negative Hessian K_h
        ab[0, 1:] = od
        new_ht = ht + solveh_banded(ab, grad)
        max_norm = np.max(np.abs(new_ht - ht))
        ht = new_ht
    h_hat = ht
    Kdm = ab[1, :].copy()   # main diagonal of K_h at the mode

    # step 2: construct the Gaussian approximation N(h_hat, K_h^{-1})
    U = cholesky_banded(ab)   # K_h = U'U with U upper banded
    logdetKh = 2*np.sum(np.log(U[1, :]))

    # log posterior kernel (normalizing constant omitted); the quadratic
    # form x'*HiSH*x is evaluated through the AR(1) innovations
    def logp(x):
        qf = ((1 - phi**2)*x[0]**2 + np.sum((x[1:] - phi*x[:-1])**2))/sigh2
        return -0.5*qf - n/2*np.sum(x) - 0.5*np.sum(np.exp(-x)*s2)

    # log Gaussian proposal density g(h)
    def logg(x):
        v = x - h_hat
        qf = np.sum(Kdm*v**2) + 2*np.sum(od*v[:-1]*v[1:])
        return -0.5*T*np.log(2*np.pi) + 0.5*logdetKh - 0.5*qf

    # step 3: choose c = kappa * p(h_hat)/g(h_hat)
    logc = np.log(kappa) + logp(h_hat) - logg(h_hat)

    # proposal kernel: q(h) \propto min{p(h)/(c g(h)), 1} g(h)
    def logq(x):
        return min(logp(x) - logc - logg(x), 0) + logg(x)

    # step 4: acceptance-rejection screening from g
    accepted_AR = False
    while not accepted_AR:
        hc = h_hat + solve_banded((0, 1), U, np.random.randn(T))  # draw from g
        if np.log(np.random.rand()) < min(logp(hc) - logc - logg(hc), 0):
            accepted_AR = True

    # step 5: Metropolis-Hastings correction
    log_alpha = logp(hc) - logp(h) + logq(h) - logq(hc)
    if np.log(np.random.rand()) < log_alpha:
        h = hc
        accept = 1
    return h, accept
