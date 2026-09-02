# vardec_SFM.py
# Computes posterior mean variance decompositions for the static factor model.
# For each draw it reconstructs the loading matrix A, forms the factor-specific
# variance contributions a_{ij}^2 * omega_j^2 and the idiosyncratic variance
# sigma_i^2, and converts them into shares of each series' total variance.
#
# Inputs:
#   store_a      : posterior draws of the free loadings, nsim x na
#   store_sig2   : posterior draws of diag(Sigma), nsim x n
#   store_omega2 : posterior draws of diag(Omega), nsim x r
#
# Outputs:
#   vd_mean   : posterior mean variance shares, n x (r+1); columns 1..r are the
#               factor shares and column r+1 is the idiosyncratic share
#   sys_mean  : posterior mean share due to all common factors, n-vector
#   idio_mean : posterior mean idiosyncratic share, n-vector

import numpy as np


def vardec_SFM(store_a, store_sig2, store_omega2):
    nsim = store_a.shape[0]
    n = store_sig2.shape[1]
    r = store_omega2.shape[1]
    vd_store = np.zeros((nsim, n, r+1))

    for isim in range(nsim):
        a = store_a[isim, :]
        sig2 = store_sig2[isim, :]
        omega2 = store_omega2[isim, :]

        # reconstruct lower-triangular loading matrix with ones on diagonal
        A = np.vstack((np.eye(r), np.zeros((n-r, r))))
        count_a = 0
        for ii in range(1, n):
            nai = min(ii, r)
            A[ii, :nai] = a[count_a:count_a+nai]
            count_a = count_a + nai

        # factor-specific variance contributions: n x r
        fac_var = A**2 * omega2

        # idiosyncratic variance contribution: n-vector
        idio_var = sig2

        # total variance of each series: n-vector
        total_var = np.sum(fac_var, axis=1) + idio_var

        # variance shares
        vd_store[isim, :, :r] = fac_var / total_var[:, None]
        vd_store[isim, :, r] = idio_var / total_var

    vd_mean = np.mean(vd_store, axis=0)
    sys_mean = np.sum(vd_mean[:, :r], axis=1)
    idio_mean = vd_mean[:, r]
    return vd_mean, sys_mean, idio_mean
