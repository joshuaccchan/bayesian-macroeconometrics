# sample_nu_griddy.py
# Griddy-Gibbs update for the degrees-of-freedom parameter nu in a
# Student-t regression model with latent scale variables lambda. The
# conditional density p(nu | lambda) is supported on (2, nu_ub).

import numpy as np
from scipy.special import gammaln
from griddy_gibbs import griddy_gibbs


def sample_nu_griddy(lam, nu_ub, n_grid):
    """Draw from p(nu | lambda) via Griddy-Gibbs.

    Inputs:
      lam   : T-vector of latent scale variables
      nu_ub : upper bound for nu
      n_grid: number of grid points

    Output:
      nu    : draw from p(nu | lambda)
    """
    T = len(lam)
    sum_loglam = np.sum(np.log(lam))
    sum_ilam = np.sum(1/lam)

    # log kernel of p(nu | lambda)
    log_kernel = lambda x: T*((x/2)*np.log(x/2) - gammaln(x/2)) \
        - (x/2 + 1)*sum_loglam - (x/2)*sum_ilam

    nu, _, _ = griddy_gibbs(log_kernel, 2, nu_ub, n_grid)
    return nu
