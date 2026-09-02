"""pred_largeVAR_FSV.py
This function computes the one-step-ahead posterior predictive means
and log predictive likelihoods of the target variables from the
large VAR with factor stochastic volatility (Section 14.4),
    y_t = A' x_t + L*f_t + u_t,
    u_t ~ N(0, D_t),   f_t ~ N(0, G_t),
    D_t = diag(exp(h_{1t}), ..., exp(h_{nt})),
    G_t = diag(exp(h_{n+1,t}), ..., exp(h_{n+r,t})),
so that the error covariance matrix is Sig_t = L*G_t*L' + D_t. The
loading matrix L is left unrestricted; the factors and loadings are
then not separately identified, but Sig_t is, and it is all the
predictive density requires. The r factor log-volatilities follow
zero-mean stationary AR(1) processes (their scales are absorbed into
the loadings), while the n idiosyncratic log-volatilities follow
stationary AR(1) processes with free levels mu_i.

The priors on the VAR coefficients ('minn', 'hs', 'mahp') are as
in pred_largeVAR_SV.py; each free loading has the prior N(0, 1).

Inputs:
  Y:          T x n matrix of observations
  Y0:         p0 x n matrix of pre-sample observations (p0 >= p)
  Tt:         scalar; forecast origin (estimation uses Y[:Tt, :], the
              forecast target is Y[Tt, :])
  p:          scalar; lag order
  r:          scalar; number of factors
  targets:    length-q vector of (0-based) indices of the target variables
  prior_type: 'minn', 'hs', or 'mahp'
  kappa1:     scalar; intercept prior variance
  kappa2:     scalar; own-lag tightness ('minn') or initial value
  kappa3:     scalar; cross-lag tightness ('minn') or initial value
  nsim:       scalar; number of post-burnin MCMC draws
  burnin:     scalar; number of burnin draws

Outputs:
  pf:        length-q posterior predictive means of the targets
  lpl_joint: scalar; log of the joint predictive density of the q
             targets at their realized values (log-mean-exp over draws)
  lpl:       length-q marginal log predictive likelihoods of the targets
"""

import numpy as np
from scipy.linalg import (cho_solve, solve_triangular, cholesky_banded,
                          solveh_banded, solve_banded)

from Minn_indep import Minn_indep
from SVAR1 import SVAR1
from sample_SVAR1para_mu import sample_SVAR1para_mu


def pred_largeVAR_FSV(Y, Y0, Tt, p, r, targets, prior_type, kappa1, kappa2,
                      kappa3, nsim, burnin):
    n = Y.shape[1]
    k = 1 + n*p
    q = len(targets)
    Yt = Y[:Tt, :]
    yreal = Y[Tt, targets]

    # priors for the volatility parameters; the factor log-volatilities
    # (last r elements) have their levels fixed at zero
    mu0 = np.zeros(n + r)
    Vmu = 100*np.ones(n + r)
    phi0 = 0.98*np.ones(n + r)
    Vphi = 0.05**2*np.ones(n + r)
    nu_h = 3*np.ones(n + r)
    S_h = 0.1*(nu_h - 1)
    free_mu = np.concatenate((np.ones(n, dtype=bool), np.zeros(r, dtype=bool)))
    V_l = 1   # prior variance of each free loading

    # construct the regressor matrix Z = [1, y_{t-1}', ..., y_{t-p}']
    tmpY = np.vstack((Y0[-p:, :], Yt))
    Z = np.zeros((Tt, n*p))
    for i in range(1, p + 1):
        Z[:, (i-1)*n:i*n] = tmpY[p-i:len(tmpY)-i, :]
    Z = np.column_stack((np.ones(Tt), Z))

    # Minnesota structure: beta0 = 0 and the constants C_ij (V_Minn with
    # unit tightness); is_own/is_cross flag own- and cross-lag positions
    beta0, V_alpha, _, _ = Minn_indep(p, kappa1, kappa2, kappa3, Y0, Yt, 0)
    _, C_Minn, _, _ = Minn_indep(p, kappa1, 1, 1, Y0, Yt, 0)
    is_int = np.zeros(n*k, dtype=bool)
    is_own = np.zeros(n*k, dtype=bool)
    for i in range(n):
        is_int[i*k] = True
        for l in range(p):
            is_own[i*k + 1 + l*n + i] = True
    is_cross = ~is_int & ~is_own
    nlag = n*k - n   # number of lag coefficients

    # regressor row for forecasting Y[Tt, :]
    xtp1 = np.concatenate(([1.0], Yt[-1:-p-1:-1, :].ravel()))

    # initialize the Markov chain
    A = np.zeros((k, n))
    mu = np.zeros(n + r)
    ZZ = Z.T @ Z
    for i in range(n):
        iVai = 1/V_alpha[i*k:(i+1)*k]
        Ki = ZZ.copy()
        Ki[np.diag_indices(k)] += iVai
        A[:, i] = np.linalg.solve(Ki, Z.T @ Yt[:, i])
        mu[i] = np.log(np.mean((Yt[:, i] - Z @ A[:, i])**2)/2)
    h = np.column_stack((np.tile(mu[:n], (Tt, 1)), np.zeros((Tt, r))))
    phi = phi0.copy()
    sigh2 = 0.05*np.ones(n + r)
    L = np.vstack((np.eye(r), 0.1*np.ones((n - r, r))))
    F = np.zeros((Tt, r))
    # global-local prior states
    tau2 = np.ones(n*k)
    nu_tau = np.ones(n*k)
    theta2 = 0.01
    xi_theta = 1
    xi_k2 = 1
    xi_k3 = 1

    store_lden = np.zeros((nsim, q + 1))
    store_mu = np.zeros((nsim, q))
    for isim in range(nsim + burnin):
        # update the prior covariance of beta under the global-local priors
        if prior_type == 'hs':
            V_alpha[~is_int] = theta2*tau2[~is_int]
        elif prior_type == 'mahp':
            V_alpha[is_own] = kappa2*tau2[is_own]*C_Minn[is_own]
            V_alpha[is_cross] = kappa3*tau2[is_cross]*C_Minn[is_cross]

        # sample the factors f_1, ..., f_T jointly via the precision sampler:
        # with Xf = kron(I_T, L), the precision Kf = diag(exp(-h_f)) +
        # Xf' D^{-1} Xf is block diagonal with T blocks L' D_t^{-1} L +
        # G_t^{-1}, i.e., banded with bandwidth r-1; it is assembled and
        # factored in banded storage rather than as a sparse matrix
        Eres = Yt - Z @ A
        W = np.exp(-h[:, :n])            # Tt x n idiosyncratic precisions
        G = np.exp(-h[:, n:])            # Tt x r factor precisions
        Kblocks = np.einsum('ji,tj,jl->til', L, W, L)   # Tt x r x r blocks
        Kblocks[:, np.arange(r), np.arange(r)] += G
        ab = np.zeros((r, Tt*r))         # upper banded storage, u = r-1
        for d in range(r):
            Ad = np.zeros(Tt*r)          # d-th superdiagonal of Kf
            for m in range(d, r):
                Ad[m::r] = Kblocks[:, m-d, m]
            ab[r-1-d, :] = Ad
        bf = ((W*Eres) @ L).ravel()      # Xf' D^{-1} e, stacked by period
        f_hat = solveh_banded(ab, bf)
        Uf = cholesky_banded(ab)         # Kf = Uf'Uf with Uf upper banded
        f = f_hat + solve_banded((0, r-1), Uf, np.random.randn(Tt*r))
        F = f.reshape(Tt, r)

        # sample (beta_i, l_i) equation by equation: given the factors, the
        # n equations are independent Gaussian regressions
        Zi = np.column_stack((Z, F))
        for i in range(n):
            wi = np.exp(-h[:, i])
            iVti = np.concatenate((1/V_alpha[i*k:(i+1)*k], np.ones(r)/V_l))
            ti0 = np.concatenate((beta0[i*k:(i+1)*k], np.zeros(r)))
            Kti = Zi.T @ (wi[:, None]*Zi)
            Kti[np.diag_indices(k + r)] += iVti
            CKti = np.linalg.cholesky(Kti)
            ti_hat = cho_solve((CKti, True), iVti*ti0 + Zi.T @ (wi*Yt[:, i]))
            ti = ti_hat + solve_triangular(CKti.T, np.random.randn(k + r),
                                           lower=False)
            A[:, i] = ti[:k]
            L[i, :] = ti[k:]

        # sample the log-volatilities series by series
        U = Yt - Z @ A - F @ L.T
        ystar = np.log(np.column_stack((U, F))**2 + 1e-4)
        for i in range(n + r):
            h[:, i] = SVAR1(ystar[:, i], h[:, i], mu[i], phi[i], sigh2[i])

        # sample the volatility parameters (mu, phi, sigh2)
        mu, phi, sigh2 = sample_SVAR1para_mu(h, mu, phi, sigh2, mu0, Vmu,
                                             phi0, Vphi, nu_h, S_h, free_mu)

        # sample the global-local prior hyperparameters
        # np.random.gamma uses (shape, scale)
        alp = A.flatten(order='F')
        if prior_type == 'hs':
            b2 = alp[~is_int]**2
            tau2[~is_int] = 1/np.random.gamma(1, 1/(1/nu_tau[~is_int]
                                                    + b2/(2*theta2)))
            nu_tau[~is_int] = 1/np.random.gamma(1, 1/(1 + 1/tau2[~is_int]))
            theta2 = 1/np.random.gamma((nlag+1)/2, 1/(1/xi_theta
                                                      + np.sum(b2/tau2[~is_int])/2))
            xi_theta = 1/np.random.gamma(1, 1/(1 + 1/theta2))
        elif prior_type == 'mahp':
            kap = kappa2*is_own + kappa3*is_cross
            b2 = alp[~is_int]**2
            denom = kap[~is_int]*C_Minn[~is_int]
            tau2[~is_int] = 1/np.random.gamma(1, 1/(1/nu_tau[~is_int]
                                                    + b2/(2*denom)))
            nu_tau[~is_int] = 1/np.random.gamma(1, 1/(1 + 1/tau2[~is_int]))
            so = np.sum(alp[is_own]**2/(tau2[is_own]*C_Minn[is_own]))
            sc = np.sum(alp[is_cross]**2/(tau2[is_cross]*C_Minn[is_cross]))
            kappa2 = 1/np.random.gamma((n*p+1)/2, 1/(1/xi_k2 + so/2))
            xi_k2 = 1/np.random.gamma(1, 1/(1 + 1/kappa2))
            kappa3 = 1/np.random.gamma((n*(n-1)*p+1)/2, 1/(1/xi_k3 + sc/2))
            xi_k3 = 1/np.random.gamma(1, 1/(1 + 1/kappa3))

        if isim >= burnin:
            isave = isim - burnin
            # forecast h_{Tt+1} via the AR(1) and the implied Sig_{Tt+1}
            htp1 = (mu + phi*(h[-1, :] - mu)
                    + np.sqrt(sigh2)*np.random.randn(n + r))
            mu_full = xtp1 @ A
            mu_q = mu_full[targets]
            Sig_tp1 = np.diag(np.exp(htp1[:n])) + L @ (np.exp(htp1[n:])[:, None]*L.T)
            Sig_q = Sig_tp1[np.ix_(targets, targets)]
            Sig_q = (Sig_q + Sig_q.T)/2
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
