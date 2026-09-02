# SFM_CE.py
# Estimates the log marginal likelihood of the static factor model using the
# cross-entropy method. The importance density is a product of a Gaussian
# for the free loadings and inverse-gamma densities for the variances,
# with parameters matched to the posterior draws. The point estimate and
# its numerical standard error are obtained from 20 importance batches.
#
# Requires: logintlike_SFM.py, lmvnpdf.py, ligampdf.py
#
# Inputs:
#   store_a     : posterior draws of the free loadings, nsim x na
#   store_Sig   : posterior draws of the idiosyncratic variances, nsim x n
#   store_Omega : posterior draws of the factor variances, nsim x r
#   Y           : data, T x n
#   prior       : function evaluating the log prior density
#   R           : number of importance samples
#
# Outputs:
#   logml     : estimated log marginal likelihood
#   logml_std : numerical standard error based on 20 importance batches

import numpy as np
from scipy.stats import gamma as gamma_dist

from logintlike_SFM import logintlike_SFM
from lmvnpdf import lmvnpdf
from ligampdf import ligampdf


def SFM_CE(store_a, store_Sig, store_Omega, Y, prior, R):
    R = 20*int(np.ceil(R/20))   # make R divisible by 20 for batching
    na = store_a.shape[1]
    n = Y.shape[1]
    r = store_Omega.shape[1]

    # estimate the parameters of the importance density
    a_bar = np.mean(store_a, axis=0)
    Da_bar = np.cov(store_a, rowvar=False)
    Da_bar = Da_bar + 1e-10*np.eye(na)   # small ridge for numerical stability
    CDa_bar = np.linalg.cholesky(Da_bar)

    # maximum likelihood fits of the gamma density to the precisions
    # (scipy.stats.gamma.fit with floc=0 returns (shape, loc, scale))
    nusig2_bar = np.zeros(n)
    Ssig2_bar = np.zeros(n)
    for ii in range(n):
        shape, _, scale = gamma_dist.fit(1/store_Sig[:, ii], floc=0)
        nusig2_bar[ii] = shape
        Ssig2_bar[ii] = 1/scale

    nuomega2_bar = np.zeros(r)
    Somega2_bar = np.zeros(r)
    for jj in range(r):
        shape, _, scale = gamma_dist.fit(1/store_Omega[:, jj], floc=0)
        nuomega2_bar[jj] = shape
        Somega2_bar[jj] = 1/scale

    # draw from the importance density
    a_IS = a_bar + (CDa_bar @ np.random.randn(na, R)).T

    Sig_IS = np.zeros((R, n))
    for ii in range(n):
        Sig_IS[:, ii] = 1/np.random.gamma(nusig2_bar[ii], 1/Ssig2_bar[ii], R)

    Omega_IS = np.zeros((R, r))
    for jj in range(r):
        Omega_IS[:, jj] = 1/np.random.gamma(nuomega2_bar[jj], 1/Somega2_bar[jj], R)

    # log importance density
    def g_IS(ax, s, o):
        return (lmvnpdf(ax, a_bar, Da_bar)
                + np.sum(ligampdf(s, nusig2_bar, Ssig2_bar))
                + np.sum(ligampdf(o, nuomega2_bar, Somega2_bar)))

    # log importance weights
    store_w = np.zeros(R)
    for isim in range(R):
        a = a_IS[isim, :]
        Sig = Sig_IS[isim, :]
        Omega = Omega_IS[isim, :]

        llike = logintlike_SFM(Y, a, Sig, Omega)
        store_w[isim] = llike + prior(a, Sig, Omega) - g_IS(a, Sig, Omega)

    # batch estimate of log marginal likelihood
    shortw = store_w.reshape(R//20, 20, order='F')   # 20 batches
    maxw = np.max(shortw, axis=0)                    # batch-specific maxima
    bigml = np.log(np.mean(np.exp(shortw - maxw), axis=0)) + maxw

    logml = np.mean(bigml)
    logml_std = np.std(bigml, ddof=1)/np.sqrt(20)

    return logml, logml_std
