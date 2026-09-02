"""pred_largeVAR_NCP.py
This function computes the one-step-ahead posterior predictive means
and log predictive likelihoods of the target variables from the
homoskedastic VAR
    y_t = A' x_t + eps_t,   eps_t ~ N(0, Sig),
with the natural conjugate prior (A, Sig) ~ NIW(A0, VA, nu0, S0)
elicited Minnesota-style by Minn_NCP.py. The NIW posterior is
available in closed form, so each iteration draws (A, Sig) directly
from the posterior and evaluates the predictive density; the log
predictive likelihoods are then formed by averaging over the draws.

Inputs:
  Y:        T x n matrix of observations
  Y0:       p0 x n matrix of pre-sample observations (p0 >= p)
  Tt:       scalar; forecast origin (estimation uses Y[:Tt, :], the
            forecast target is Y[Tt, :])
  p:        scalar; lag order
  targets:  length-q vector of (0-based) indices of the target variables
  kappa1:   scalar; prior variance on intercepts
  kappa2:   scalar; overall shrinkage on lag coefficients
  nsim:     scalar; number of posterior draws

Outputs:
  pf:        length-q posterior predictive means of the targets
  lpl_joint: scalar; log of the joint predictive density of the q
             targets at their realized values (log-mean-exp over draws)
  lpl:       length-q marginal log predictive likelihoods of the targets
"""

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.stats import invwishart

from Minn_NCP import Minn_NCP


def pred_largeVAR_NCP(Y, Y0, Tt, p, targets, kappa1, kappa2, nsim):
    n = Y.shape[1]
    k = 1 + n*p
    q = len(targets)
    Yt = Y[:Tt, :]
    yreal = Y[Tt, targets]

    # construct the regressor matrix Z = [1, y_{t-1}', ..., y_{t-p}']
    tmpY = np.vstack((Y0[-p:, :], Yt))
    Z = np.zeros((Tt, n*p))
    for i in range(1, p + 1):
        Z[:, (i-1)*n:i*n] = tmpY[p-i:len(tmpY)-i, :]
    Z = np.column_stack((np.ones(Tt), Z))

    # natural conjugate prior and analytical posterior
    A0, VA, nu0, S0 = Minn_NCP(Yt, Y0, p, kappa1, kappa2, 0)
    iva = 1/np.diag(VA)   # diagonal of VA^{-1}
    KA = np.diag(iva) + Z.T @ Z
    CKA = np.linalg.cholesky(KA)
    A_hat = cho_solve((CKA, True), iva[:, None]*A0 + Z.T @ Yt)
    S_hat = S0 + A0.T @ (iva[:, None]*A0) + Yt.T @ Yt - A_hat.T @ KA @ A_hat
    S_hat = (S_hat + S_hat.T)/2   # adjust for rounding errors
    nu_hat = nu0 + Tt

    # regressor row for forecasting Y[Tt, :]
    xtp1 = np.concatenate(([1.0], Yt[-1:-p-1:-1, :].ravel()))

    store_lden = np.zeros((nsim, q + 1))
    store_mu = np.zeros((nsim, q))
    for isim in range(nsim):
        # sample Sig and A from the NIW posterior
        Sig = invwishart.rvs(df=nu_hat, scale=S_hat)
        CSig = np.linalg.cholesky(Sig)
        A = A_hat + solve_triangular(CKA.T, np.random.randn(k, n),
                                     lower=False) @ CSig.T

        # evaluate the predictive density at the realized targets
        mu_full = xtp1 @ A
        mu_q = mu_full[targets]
        Sig_q = Sig[np.ix_(targets, targets)]
        CSig_q = np.linalg.cholesky(Sig_q)
        u = solve_triangular(CSig_q, yreal - mu_q, lower=True)
        lden_joint = (-q/2*np.log(2*np.pi) - np.sum(np.log(np.diag(CSig_q)))
                      - 0.5*(u @ u))
        dSig_q = np.diag(Sig_q)
        lden = -0.5*np.log(2*np.pi*dSig_q) - 0.5*(yreal - mu_q)**2/dSig_q
        store_mu[isim, :] = mu_q
        store_lden[isim, :] = np.concatenate((lden, [lden_joint]))

    # aggregate across draws
    pf = store_mu.mean(axis=0)
    tmpmax = store_lden.max(axis=0)
    lpls = np.log(np.mean(np.exp(store_lden - tmpmax), axis=0)) + tmpmax
    lpl = lpls[:q]
    lpl_joint = lpls[q]
    return pf, lpl_joint, lpl
