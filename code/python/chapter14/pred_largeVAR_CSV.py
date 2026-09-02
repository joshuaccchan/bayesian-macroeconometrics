"""pred_largeVAR_CSV.py
This function computes the one-step-ahead posterior predictive means
and log predictive likelihoods of the target variables from the VAR
with common stochastic volatility (Section 14.2),
    y_t = A' x_t + eps_t,   eps_t ~ N(0, exp(h_t)*Sig),
    h_t = phi*h_{t-1} + u_t^h,   u_t^h ~ N(0, sigh2),
    h_1 ~ N(0, sigh2/(1 - phi^2)),
with the natural conjugate prior (A, Sig) ~ NIW(A0, VA, nu0, S0)
elicited Minnesota-style by Minn_NCP.py.

Inputs:
  Y:        T x n matrix of observations
  Y0:       p0 x n matrix of pre-sample observations (p0 >= p)
  Tt:       scalar; forecast origin (estimation uses Y[:Tt, :], the
            forecast target is Y[Tt, :])
  p:        scalar; lag order
  targets:  length-q vector of (0-based) indices of the target variables
  kappa1:   scalar; prior variance on intercepts
  kappa2:   scalar; overall shrinkage on lag coefficients
  nsim:     scalar; number of post-burnin MCMC draws
  burnin:   scalar; number of burnin draws

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
from sample_CSV_h_ARMH import sample_CSV_h_ARMH


def pred_largeVAR_CSV(Y, Y0, Tt, p, targets, kappa1, kappa2, nsim, burnin):
    n = Y.shape[1]
    k = 1 + n*p
    q = len(targets)
    Yt = Y[:Tt, :]
    yreal = Y[Tt, targets]

    # priors for the volatility parameters
    phi0 = 0.98
    Vphi = 0.05**2                    # phi ~ TN_(-1,1)(phi0, Vphi)
    nu_h = 3
    S_h = 0.1*(nu_h - 1)              # sigh2 ~ IG(nu_h, S_h)

    # construct the regressor matrix Z = [1, y_{t-1}', ..., y_{t-p}']
    tmpY = np.vstack((Y0[-p:, :], Yt))
    Z = np.zeros((Tt, n*p))
    for i in range(1, p + 1):
        Z[:, (i-1)*n:i*n] = tmpY[p-i:len(tmpY)-i, :]
    Z = np.column_stack((np.ones(Tt), Z))

    # natural conjugate prior
    A0, VA, nu0, S0 = Minn_NCP(Yt, Y0, p, kappa1, kappa2, 0)
    iva = 1/np.diag(VA)   # diagonal of VA^{-1}

    # regressor row for forecasting Y[Tt, :]
    xtp1 = np.concatenate(([1.0], Yt[-1:-p-1:-1, :].ravel()))

    # initialize the Markov chain
    phi = phi0
    sigh2 = 0.05
    h = np.zeros(Tt)

    store_lden = np.zeros((nsim, q + 1))
    store_mu = np.zeros((nsim, q))
    for isim in range(nsim + burnin):
        # sample Sig and A given h with Omega^{-1} = diag(exp(-h))
        iOm = np.exp(-h)
        ZiOm = Z.T*iOm
        KA = np.diag(iva) + ZiOm @ Z
        CKA = np.linalg.cholesky(KA)
        A_hat = cho_solve((CKA, True), iva[:, None]*A0 + ZiOm @ Yt)
        S_hat = (S0 + A0.T @ (iva[:, None]*A0) + Yt.T @ (iOm[:, None]*Yt)
                 - A_hat.T @ KA @ A_hat)
        S_hat = (S_hat + S_hat.T)/2   # adjust for rounding errors
        Sig = invwishart.rvs(df=nu0 + Tt, scale=S_hat)
        CSig = np.linalg.cholesky(Sig)
        A = A_hat + solve_triangular(CKA.T, np.random.randn(k, n),
                                     lower=False) @ CSig.T

        # sample the common log-volatility h via the Laplace-based ARMH step
        U = Yt - Z @ A
        tmp = solve_triangular(CSig, U.T, lower=True).T  # standardized resid
        s2 = np.sum(tmp**2, axis=1)
        h, _ = sample_CSV_h_ARMH(s2, phi, sigh2, h, n, 30)

        # sample sigh2
        eh = np.concatenate(([h[0]*np.sqrt(1 - phi**2)], h[1:] - phi*h[:-1]))
        # np.random.gamma uses (shape, scale)
        sigh2 = 1/np.random.gamma(nu_h + Tt/2, 1/(S_h + np.sum(eh**2)/2))

        # sample phi via an independence-chain MH step
        Kphi = 1/Vphi + np.sum(h[:-1]**2)/sigh2
        phihat = (phi0/Vphi + h[:-1] @ h[1:]/sigh2)/Kphi
        phic = phihat + np.random.randn()/np.sqrt(Kphi)

        def gphi(x):
            return 0.5*np.log(1 - x**2) - 0.5*(1 - x**2)/sigh2*h[0]**2
        if np.abs(phic) < 0.998 and np.exp(gphi(phic) - gphi(phi)) > np.random.rand():
            phi = phic

        if isim >= burnin:
            isave = isim - burnin
            # forecast h_{Tt+1} via the AR(1) and the implied Sig_{Tt+1}
            htp1 = phi*h[-1] + np.sqrt(sigh2)*np.random.randn()
            mu_full = xtp1 @ A
            mu_q = mu_full[targets]
            Sig_q = np.exp(htp1)*Sig[np.ix_(targets, targets)]
            CSig_q = np.linalg.cholesky(Sig_q)
            u = solve_triangular(CSig_q, yreal - mu_q, lower=True)
            lden_joint = (-q/2*np.log(2*np.pi)
                          - np.sum(np.log(np.diag(CSig_q))) - 0.5*(u @ u))
            dSig_q = np.diag(Sig_q)
            lden = -0.5*np.log(2*np.pi*dSig_q) - 0.5*(yreal - mu_q)**2/dSig_q
            store_mu[isave, :] = mu_q
            store_lden[isave, :] = np.concatenate((lden, [lden_joint]))

    # aggregate across draws
    pf = store_mu.mean(axis=0)
    tmpmax = store_lden.max(axis=0)
    lpls = np.log(np.mean(np.exp(store_lden - tmpmax), axis=0)) + tmpmax
    lpl = lpls[:q]
    lpl_joint = lpls[q]
    return pf, lpl_joint, lpl
