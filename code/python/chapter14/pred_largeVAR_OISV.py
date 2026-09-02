"""pred_largeVAR_OISV.py
This function computes the one-step-ahead posterior predictive means
and log predictive likelihoods of the target variables from the
large VAR with order-invariant stochastic volatility (Section 14.3),
    y_t = A' x_t + eps_t,    eps_t ~ N(0, Sig_t),
    Sig_t^{-1} = B0' D_t^{-1} B0,
    D_t = diag(exp(h_{1t}), ..., exp(h_{nt})),
where B0 is an unrestricted (dense) n x n matrix and each
log-volatility follows the zero-mean stationary AR(1) process
    h_{it} = phi_i*h_{i,t-1} + u_{it}^h,   u_{it}^h ~ N(0, sigh2_i),
    h_{i1} ~ N(0, sigh2_i/(1 - phi_i^2)).

The priors on the VAR coefficients ('minn', 'hs', 'mahp') are as in
pred_largeVAR_SV.py; each element of B0 has the prior N(b0_ij, 10)
with prior mean the identity matrix.

Inputs:
  Y:          T x n matrix of observations
  Y0:         p0 x n matrix of pre-sample observations (p0 >= p)
  Tt:         scalar; forecast origin (estimation uses Y[:Tt, :], the
              forecast target is Y[Tt, :])
  p:          scalar; lag order
  targets:    length-q vector of (0-based) indices of the target variables
  prior_type: 'minn', 'minnH', 'hs', or 'mahp' (see pred_largeVAR_SV.py)
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
  hyp:       nsim x 3 saved hyperparameter draws [kappa2, kappa3,
             theta2] (diagnostics): kappa2/kappa3 are the sampled
             tightness under 'minnH'/'mahp', theta2 the horseshoe global
             scale under 'hs'; unused entries stay at their input values
  draws_mu:  nsim x q predictive-mean draws, for comparing the mean and
             median point forecast
"""

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

from Minn_indep import Minn_indep
from SVAR1 import SVAR1
from sample_SVAR1para import sample_SVAR1para
from sample_B0 import sample_B0


def pred_largeVAR_OISV(Y, Y0, Tt, p, targets, prior_type, kappa1, kappa2,
                       kappa3, nsim, burnin):
    n = Y.shape[1]
    k = 1 + n*p
    q = len(targets)
    Yt = Y[:Tt, :]
    yreal = Y[Tt, targets]

    # priors for the volatility parameters and B0
    phi0 = 0.95
    Vphi = 0.05**2
    nu_h = 3*np.ones(n)
    S_h = 0.05*(nu_h - 1)
    b0_B0 = np.eye(n)
    iV_B0 = np.eye(n)/10

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
    ZZ = Z.T @ Z
    for i in range(n):
        iVai = 1/V_alpha[i*k:(i+1)*k]
        Ki = ZZ.copy()
        Ki[np.diag_indices(k)] += iVai
        A[:, i] = np.linalg.solve(Ki, Z.T @ Yt[:, i])
    U = Yt - Z @ A
    Sig_hat = U.T @ U/Tt
    B0 = np.diag(1/np.sqrt(np.diag(Sig_hat)))
    h = np.zeros((Tt, n))
    phi = phi0*np.ones(n)
    sigh2 = 0.05*np.ones(n)
    # global-local prior states
    tau2 = np.ones(n*k)
    nu_tau = np.ones(n*k)
    theta2 = 0.01
    xi_theta = 1
    xi_k2 = 1
    xi_k3 = 1

    store_lden = np.zeros((nsim, q + 1))
    store_mu = np.zeros((nsim, q))
    store_hyp = np.zeros((nsim, 3))   # [kappa2, kappa3, theta2] per saved draw
    for isim in range(nsim + burnin):
        # update the prior covariance of beta under the global-local priors
        if prior_type == 'hs':
            V_alpha[~is_int] = theta2*tau2[~is_int]
        elif prior_type == 'minnH':
            V_alpha[is_own] = kappa2*C_Minn[is_own]
            V_alpha[is_cross] = kappa3*C_Minn[is_cross]
        elif prior_type == 'mahp':
            V_alpha[is_own] = kappa2*tau2[is_own]*C_Minn[is_own]
            V_alpha[is_cross] = kappa3*tau2[is_cross]*C_Minn[is_cross]

        # sample B0 row by row via the Waggoner-Zha-Villani scheme
        U = Yt - Z @ A
        B0 = sample_B0(B0, U, h, b0_B0, iV_B0)

        # sample beta equation by equation (Corollary 14.1)
        eh_inv = np.exp(-h)
        for i in range(n):
            A[:, i] = 0
            Etil = (Yt - Z @ A) @ B0.T  # row t is z_t' = (B0(y_t - A_{-i}'x_t))'
            bi = B0[:, i]
            w = eh_inv @ bi**2          # w_it = sum_j B0[j,i]^2 exp(-h_jt)
            c = (eh_inv*Etil) @ bi
            iVai = 1/V_alpha[i*k:(i+1)*k]
            Kai = Z.T @ (w[:, None]*Z)
            Kai[np.diag_indices(k)] += iVai
            CKai = np.linalg.cholesky(Kai)
            ai_hat = cho_solve((CKai, True),
                               iVai*beta0[i*k:(i+1)*k] + Z.T @ c)
            A[:, i] = ai_hat + solve_triangular(CKai.T, np.random.randn(k),
                                                lower=False)

        # sample the log-volatilities equation by equation; each h_i is
        # recentered to mean zero, since its level is absorbed into the
        # scale of the corresponding row of B0
        Eorth = (Yt - Z @ A) @ B0.T
        ystar = np.log(Eorth**2 + 1e-4)
        for i in range(n):
            h[:, i] = SVAR1(ystar[:, i], h[:, i], 0, phi[i], sigh2[i])
            h[:, i] = h[:, i] - np.mean(h[:, i])

        # sample the volatility parameters (phi, sigh2)
        phi, sigh2 = sample_SVAR1para(h, phi, sigh2, phi0, Vphi, nu_h, S_h)

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
        elif prior_type == 'minnH':
            so = np.sum(alp[is_own]**2/C_Minn[is_own])
            sc = np.sum(alp[is_cross]**2/C_Minn[is_cross])
            kappa2 = 1/np.random.gamma((n*p+1)/2, 1/(1/xi_k2 + so/2))
            xi_k2 = 1/np.random.gamma(1, 1/(1 + 1/kappa2))
            kappa3 = 1/np.random.gamma((n*(n-1)*p+1)/2, 1/(1/xi_k3 + sc/2))
            xi_k3 = 1/np.random.gamma(1, 1/(1 + 1/kappa3))
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
            htp1 = phi*h[-1, :] + np.sqrt(sigh2)*np.random.randn(n)
            mu_full = xtp1 @ A
            mu_q = mu_full[targets]
            Sig_tp1 = np.linalg.solve(
                B0, np.linalg.solve(B0, np.diag(np.exp(htp1))).T).T
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
            store_hyp[isave, :] = [kappa2, kappa3, theta2]

    # aggregate across draws
    pf = store_mu.mean(axis=0)
    tmpmax = store_lden.max(axis=0)
    lpls = np.log(np.mean(np.exp(store_lden - tmpmax), axis=0)) + tmpmax
    lpl = lpls[:q]
    lpl_joint = lpls[q]
    hyp = store_hyp        # saved [kappa2 kappa3 theta2] draws (diagnostics)
    draws_mu = store_mu    # nsim x q predictive-mean draws (mean vs median)
    return pf, lpl_joint, lpl, hyp, draws_mu
