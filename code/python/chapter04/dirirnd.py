# dirirnd.py
# Draws from a Dirichlet distribution D(alpha) using the standard
# gamma-normalization construction: if g_m ~ G(alpha_m, 1) iid, then
# (g_1, ..., g_M) / sum_m g_m has a D(alpha) distribution.
#
# Inputs:
#   alpha: length-M vector of concentration parameters (alpha_m > 0)
#   N    : number of draws
#
# Output:
#   W    : N-by-M matrix; each row is an independent draw on the
#          unit simplex

import numpy as np


def dirirnd(alpha, N):
    alpha = np.asarray(alpha, dtype=float).flatten()
    G = np.random.gamma(np.tile(alpha, (N, 1)), 1)   # N-by-M matrix
    W = G / G.sum(axis=1, keepdims=True)
    return W
