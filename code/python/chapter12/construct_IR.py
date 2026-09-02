# construct_IR.py
# Computes the structural impulse response function of a VAR(p) by iterating
# the VAR forward under two scenarios -- one in which a structural shock hits
# at the impact horizon and one without -- and differencing the two paths.
#
# Inputs:
#   beta  : nk-vector of VAR coefficients, k = 1+n*p
#   Sig   : n x n error covariance matrix
#   n_hz  : number of horizons (including impact, h = 0)
#   shock : n-vector structural shock (e.g., a unit vector e_j)
#
# Output:
#   yIR : n_hz x n impulse responses; row h gives the response at horizon h

import numpy as np


def construct_IR(beta, Sig, n_hz, shock):
    n = Sig.shape[0]
    p = (beta.size//n - 1)//n
    k = 1 + n*p
    CSig = np.linalg.cholesky(Sig)
    # kron(I_n, x_t')*beta = x_t' * A with A = reshape(beta, k, n) column-major
    A = beta.reshape(k, n, order='F')

    # initialize: shocked path starts at CSig*shock, baseline at 0
    tmpZ1 = np.zeros((p, n))
    tmpZ = np.zeros((p, n))
    Yt1 = CSig @ shock
    Yt = np.zeros(n)
    yIR = np.zeros((n_hz, n))
    yIR[0, :] = Yt1

    for t in range(1, n_hz):
        # update the lagged values for each path
        tmpZ = np.vstack((Yt, tmpZ[:-1, :]))
        tmpZ1 = np.vstack((Yt1, tmpZ1[:-1, :]))

        # shocked path: iterate the VAR forward
        Xt1 = np.concatenate(([1], tmpZ1.flatten()))
        Yt1 = Xt1 @ A

        # baseline path: iterate the VAR forward
        Xt = np.concatenate(([1], tmpZ.flatten()))
        Yt = Xt @ A

        # impulse response = difference between the two paths
        yIR[t, :] = Yt1 - Yt
    return yIR
