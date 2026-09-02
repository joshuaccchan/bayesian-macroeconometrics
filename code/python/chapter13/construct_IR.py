"""construct_IR.py
Computes the structural impulse response function of a VAR(p) by iterating
the VAR forward under two scenarios -- one in which a structural shock hits
at the impact horizon and one without -- and differencing the two paths.

Inputs:
  beta  : length-nk vector of VAR coefficients, k = 1+n*p
  Sig   : n x n error covariance matrix
  n_hz  : number of horizons (including impact, h = 0)
  shock : length-n structural shock vector (e.g., a unit vector e_j)

Output:
  yIR : n_hz x n impulse responses; row h gives the response at horizon h
"""

import numpy as np


def construct_IR(beta, Sig, n_hz, shock):
    n = Sig.shape[0]
    p = (len(beta)//n - 1)//n
    CSig = np.linalg.cholesky(Sig)

    # coefficients by equation: row i of B holds equation i's k = 1+n*p
    # coefficients, so that kron(I_n, [1, z']) @ beta == B @ [1; z]
    B = beta.reshape(n, 1 + n*p)

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
        Z1 = tmpZ1.flatten()               # reshape(tmpZ1', 1, n*p)
        Yt1 = B @ np.concatenate(([1.0], Z1))

        # baseline path: iterate the VAR forward
        Z = tmpZ.flatten()
        Yt = B @ np.concatenate(([1.0], Z))

        # impulse response = difference between the two paths
        yIR[t, :] = Yt1 - Yt
    return yIR
