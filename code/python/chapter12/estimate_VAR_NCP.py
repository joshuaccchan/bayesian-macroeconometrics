# estimate_VAR_NCP.py
# Computes the posterior hyperparameters of a VAR(p) under the natural
# conjugate normal-inverse-Wishart prior, building the regressor matrix from
# the sample Y and the pre-sample observations Y0.
#
# Inputs:
#   Y   : T x n matrix of observations
#   Y0  : p0 x n matrix of pre-sample observations (p0 >= p)
#   p   : lag order
#   A0  : k x n prior mean of the coefficient matrix A, k = 1+n*p
#   VA  : k x k prior scale matrix, so that Cov(vec(A)|Sigma) = Sigma x VA
#   nu0 : prior degrees of freedom for the inverse-Wishart on Sigma
#   S0  : n x n prior scale matrix for the inverse-Wishart on Sigma
#
# Outputs:
#   A_hat  : k x n posterior mean of A
#   KA     : k x k posterior precision matrix
#   nu_hat : posterior degrees of freedom
#   S_hat  : n x n posterior scale matrix

import numpy as np
from scipy.linalg import solve_triangular


def estimate_VAR_NCP(Y, Y0, p, A0, VA, nu0, S0):
    T, n = Y.shape
    k = 1 + n*p

    # construct the T x k regressor matrix Z whose t-th row is
    # x_t' = (1, y_{t-1}', ..., y_{t-p}')
    tmpY = np.vstack((Y0[-p:, :], Y))
    Z = np.zeros((T, n*p))
    for i in range(1, p+1):
        Z[:, (i-1)*n:i*n] = tmpY[p-i:p+T-i, :]
    Z = np.column_stack((np.ones(T), Z))

    # compute posterior hyperparameters
    iVA = np.linalg.solve(VA, np.eye(k))   # VA^{-1}
    KA = iVA + Z.T @ Z
    # Cholesky factorize KA once and obtain A_hat by two triangular solves
    CKA = np.linalg.cholesky(KA)
    A_hat = solve_triangular(CKA.T,
                             solve_triangular(CKA, iVA @ A0 + Z.T @ Y, lower=True),
                             lower=False)
    S_hat = S0 + A0.T @ iVA @ A0 + Y.T @ Y - A_hat.T @ KA @ A_hat
    # symmetrize to correct for rounding errors
    S_hat = (S_hat + S_hat.T)/2
    nu_hat = nu0 + T
    return A_hat, KA, nu_hat, S_hat
