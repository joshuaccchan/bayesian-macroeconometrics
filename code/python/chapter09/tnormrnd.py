# tnormrnd.py
# This function draws from a univariate truncated normal distribution using
# the inverse-transform method. It returns N draws from a normal
# distribution with mean mu and variance sigma2, truncated to the interval (a,b)
#
# Inputs:
# mu: mean (scalar or length-N vector)
# sigma2: variance (scalar or length-N vector)
# a: lower truncation point (scalar)
# b: upper truncation point (scalar)
# N: number of draws (optional; defaults to length of mu)
#
# Output:
# draws: length-N vector of truncated normal draws
#        (a scalar when N == 1, as MATLAB's 1-by-1 behaves as a scalar)

import numpy as np
from scipy.stats import norm


def tnormrnd(mu, sigma2, a, b, N=None):
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    sigma2 = np.atleast_1d(np.asarray(sigma2, dtype=float))

    # length of mean vector
    K = mu.size

    # if N not supplied, set equal to length of mu
    if N is None:
        N = K

    # dimension check
    if (K != N or sigma2.size != N) and K != 1:
        raise ValueError('Dimensions of mu and sigma2 must equal N.')

    # expand scalars to vectors if necessary
    if K == 1:
        mu = np.repeat(mu, N)
        sigma2 = np.repeat(sigma2, N)

    sigma = np.sqrt(sigma2)
    u = np.random.rand(N)

    # compute CDF values at truncation points
    p1 = norm.cdf((a - mu)/sigma)
    p2 = norm.cdf((b - mu)/sigma)

    # apply inverse CDF transformation
    C = norm.ppf(p1 + (p2 - p1)*u)

    # transform back to truncated normal draw
    draws = mu + sigma*C
    if N == 1:
        return draws[0]
    return draws
