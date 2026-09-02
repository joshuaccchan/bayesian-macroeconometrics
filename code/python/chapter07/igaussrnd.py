"""igaussrnd.py
Draws from the inverse Gaussian distribution IGAUSS(psi, mu) with
kernel
  f(x) \\propto x^{-3/2} exp{ -psi (x-mu)^2 / (2 x mu^2) },  x > 0.
Uses the standard chi-squared-based representation from
Michael, Schucany, and Haas (1976).

Inputs:
  psi : positive shape parameter (scalar or length-n vector)
  mu  : positive mean parameter  (scalar or length-n vector)
  n   : number of draws (defaults to max(psi.size, mu.size))

Output:
  x   : length-n vector of draws from IGAUSS(psi, mu)
"""

import numpy as np


def igaussrnd(psi, mu, n=None):
    if n is None:
        n = max(np.size(psi), np.size(mu))

    # expand scalars to length n; force 1-D vectors
    if np.isscalar(psi) or np.size(psi) == 1:
        psi = np.full(n, float(np.asarray(psi).item()))
    else:
        psi = np.asarray(psi, dtype=float).ravel()
    if np.isscalar(mu) or np.size(mu) == 1:
        mu = np.full(n, float(np.asarray(mu).item()))
    else:
        mu = np.asarray(mu, dtype=float).ravel()

    if psi.size != n or mu.size != n:
        raise ValueError('psi and mu must be scalars or vectors of the same length n.')

    # step 1: nu0 ~ chi^2_1
    nu0 = np.random.randn(n)**2

    # step 2: candidate draws
    sqrt_term = np.sqrt(4*mu*psi*nu0 + (mu**2)*(nu0**2))
    x1 = mu + (mu**2)*nu0/(2*psi) - (mu/(2*psi))*sqrt_term
    x2 = (mu**2)/x1

    # step 3: accept/reject switch
    p = mu/(mu + x1)
    U = np.random.rand(n) < p
    x = np.where(U, x1, x2)
    return x
