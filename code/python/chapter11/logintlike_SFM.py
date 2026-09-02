# logintlike_SFM.py
# Evaluates the log integrated likelihood of the static factor model, i.e., the
# Gaussian density of the data with the latent factors integrated out:
#   y_t ~ N(0, A*Omega*A' + Sigma).
# The n x n inverse is obtained via the Sherman-Morrison-Woodbury identity, and
# the log-determinant and quadratic form are computed from the Cholesky factor
# of the precision matrix, avoiding direct inversion of an n x n matrix.
#
# Inputs:
#   Y     : data, T x n
#   a     : free elements of the lower-triangular loading matrix A
#   Sig   : idiosyncratic variances, n-vector
#   Omega : factor variances, r-vector
#
# Output:
#   loglike : log integrated likelihood

import numpy as np


def logintlike_SFM(Y, a, Sig, Omega):
    T, n = Y.shape
    r = len(Omega)

    # construct unit lower-triangular A
    A = np.vstack((np.eye(r), np.zeros((n-r, r))))
    count_a = 0
    for ii in range(1, n):
        nai = min(ii, r)
        A[ii, :nai] = a[count_a:count_a+nai]
        count_a = count_a + nai

    # diagonal precision matrices (kept as vectors)
    iSig = 1/Sig
    iOmega = np.diag(1/Omega)

    # compute (A*Omega*A' + Sig)^{-1} using Woodbury identity
    AiSig = A.T * iSig                                    # r x n
    iB = np.diag(iSig) - AiSig.T @ np.linalg.solve(iOmega + AiSig @ A, AiSig)

    CiB = np.linalg.cholesky(iB)   # Cholesky factor of precision (lower)
    CY = CiB.T @ Y.T
    quad = np.sum(CY**2)   # quadratic term: sum_t y_t' iB y_t
    loglike = -0.5*T*n*np.log(2*np.pi) + T*np.sum(np.log(np.diag(CiB))) - 0.5*quad
    return loglike
