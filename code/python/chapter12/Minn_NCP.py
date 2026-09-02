# Minn_NCP.py
# Constructs the natural conjugate (normal-inverse-Wishart) prior
# hyperparameters (A, Sigma) ~ NIW(A0, VA, nu0, S0) using Minnesota-style
# elicitation, with residual variances from univariate AR(p) fits.
#
# Inputs:
#   Y      : T x n matrix of observations
#   Y0     : p0 x n matrix of pre-sample observations (p0 >= p)
#   p      : lag order
#   kappa1 : prior variance on the intercepts
#   kappa2 : overall shrinkage on the lag coefficients
#   rw     : 1 = random-walk prior mean (first own lag = 1),
#            0 = zero prior mean (for growth-rate data)
#
# Outputs:
#   A0  : k x n prior mean of the coefficient matrix A, k = 1+n*p
#   VA  : k x k diagonal prior scale matrix for vec(A)
#   nu0 : prior degrees of freedom for the inverse-Wishart
#   S0  : n x n prior scale matrix for the inverse-Wishart

import numpy as np


def Minn_NCP(Y, Y0, p, kappa1, kappa2, rw):
    T, n = Y.shape
    k = 1 + n*p

    # estimate residual variances from univariate AR(p) models
    s2 = np.zeros(n)
    for i in range(n):
        tmpY = np.concatenate((Y0[-p:, i], Y[:, i]))
        Z_ar = np.zeros((T, p))
        for j in range(1, p+1):
            Z_ar[:, j-1] = tmpY[p-j:p+T-j]
        Z_ar = np.column_stack((np.ones(T), Z_ar))
        b_ar = np.linalg.lstsq(Z_ar, Y[:, i], rcond=None)[0]
        e_ar = Y[:, i] - Z_ar @ b_ar
        s2[i] = np.mean(e_ar**2)

    # prior mean A0: k x n
    A0 = np.zeros((k, n))
    if rw:
        for j in range(n):
            A0[1 + j, j] = 1   # coefficient on y_{j,t-1} in equation j

    # prior scale matrix VA: k x k diagonal
    # intercept: kappa1
    # lag l, variable r: kappa2 / (l^2 * s_r^2)
    va = np.zeros(k)
    va[0] = kappa1   # intercept
    for l in range(1, p+1):
        for r in range(1, n+1):
            idx = (l-1)*n + r
            va[idx] = kappa2/(l**2*s2[r-1])
    VA = np.diag(va)

    # inverse-Wishart hyperparameters
    nu0 = n + 2
    S0 = np.diag(s2)
    return A0, VA, nu0, S0
