# SURform.py
# This function constructs a sparse SUR/stacked design matrix
#
# Input:
# X: T-by-k design matrix, where row t is x_t'
#
# Output:
# Z: T-by-(T*k) sparse matrix diag(x_1', ..., x_T')

import numpy as np
from scipy import sparse


def SURform(X):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError('X must be a numeric 2-D matrix.')

    T, k = X.shape

    # row indices: each t repeated k times
    row_idx = np.repeat(np.arange(T), k)

    # column indices: 0,1,...,T*k-1 (block for each t is consecutive k columns)
    col_idx = np.arange(T*k)

    # values: stack rows of X as (x_1', x_2', ..., x_T')'
    vals = X.flatten()

    Z = sparse.csc_array((vals, (row_idx, col_idx)), shape=(T, T*k))
    return Z
