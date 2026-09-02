# griddy_gibbs.py
# Generic Griddy-Gibbs sampler for a univariate target density with
# bounded support (a,b), given an unnormalized log-density logf(x).
#
# Inputs:
#   logf  : function returning log unnormalized density at x;
#           must accept a vector input and return a vector output of
#           the same size
#   a, b  : lower and upper bounds (a < b), finite
#   n_grid: number of grid points (integer >= 2)
#
# Outputs:
#   x_draw   : one draw from the Griddy-Gibbs approximation to the target
#   x_grid   : grid points used
#   logf_grid: logf evaluated on x_grid

import numpy as np


def griddy_gibbs(logf, a, b, n_grid):
    # jittered grid to avoid repeated draws
    step = (b - a) / (n_grid + 1)   # grid spacing
    x_grid = a + step*np.arange(1, n_grid+1)
    x_grid = x_grid + (np.random.rand(n_grid) - 0.5)*step   # small jitter
    # enforce strict bounds
    eps = np.finfo(float).eps
    x_grid = np.minimum(np.maximum(x_grid, a + eps), b - eps)

    # evaluate logf on the grid and normalize
    logf_grid = logf(x_grid)
    w = np.exp(logf_grid - np.max(logf_grid))
    w = w / np.sum(w)   # normalize to sum to 1

    cdf_grid = np.cumsum(w)
    u = np.random.rand()
    idx = np.argmax(cdf_grid >= u)   # first index with cdf >= u
    x_draw = x_grid[idx]
    return x_draw, x_grid, logf_grid
