"""pred_largeVAR_SV.py
This function computes the one-step-ahead posterior predictive means
and log predictive likelihoods of the target variables from the
large VAR with Cholesky stochastic volatility (Section 14.3),
    y_t = A' x_t + eps_t,    eps_t ~ N(0, Sig_t),
    Sig_t^{-1} = B0' D_t^{-1} B0,
    D_t = diag(exp(h_{1t}), ..., exp(h_{nt})),
where B0 is unit lower triangular and each log-volatility follows a
stationary AR(1) with level mu_i,
    h_{it} = mu_i + phi_i*(h_{i,t-1} - mu_i) + u_{it}^h,
    u_{it}^h ~ N(0, sigh2_i),
    h_{i1} ~ N(mu_i, sigh2_i/(1 - phi_i^2)).

Four priors on the VAR coefficients are supported (Section 14.3.1);
in each case the intercepts have fixed variance kappa1:
  'minn':  independent Minnesota prior with own-lag tightness kappa2
           and cross-lag tightness kappa3, both fixed;
  'minnH': Minnesota prior with the same structure, but with the own-
           and cross-lag tightness estimated, beta_ij ~ N(0,
           kappa_ij*C_ij) with sqrt(kappa2), sqrt(kappa3) ~ C+(0,1)
           and C_ij the Minnesota constants;
  'hs':    horseshoe prior, beta_ij ~ N(0, theta^2*tau_ij^2) with
           theta, tau_ij ~ C+(0,1);
  'mahp':  Minnesota-type global-local prior of Chan (2021),
           beta_ij ~ N(0, kappa_ij*tau_ij^2*C_ij) with
           sqrt(kappa2), sqrt(kappa3), tau_ij ~ C+(0,1). It adds the
           local scales tau_ij to 'minnH'.
For 'minnH' and 'mahp' the inputs kappa2 and kappa3 are used as
initial values. The half-Cauchy scales are updated with the auxiliary
inverse-gamma representation of Makalic and Schmidt (2016).

Inputs:
  Y:          T x n matrix of observations
  Y0:         p0 x n matrix of pre-sample observations (p0 >= p)
  Tt:         scalar; forecast origin (estimation uses Y[:Tt, :], the
              forecast target is Y[Tt, :])
  p:          scalar; lag order
  targets:    length-q vector of (0-based) indices of the target variables
  prior_type: 'minn', 'minnH', 'hs', or 'mahp'
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
from scipy.linalg import cho_solve, solve_triangular

from Minn_indep import Minn_indep
from SVAR1 import SVAR1
from sample_SVAR1para_mu import sample_SVAR1para_mu


def pred_largeVAR_SV(Y, Y0, Tt, p, targets, prior_type, kappa1, kappa2,
                     kappa3, nsim, burnin):
    n = Y.shape[1]
    k = 1 + n*p
    q = len(targets)
    Yt = Y[:Tt, :]
    yreal = Y[Tt, targets]

    # priors for the volatility parameters
    mu0 = np.zeros(n)
    Vmu = 100*np.ones(n)
    phi0 = 0.98*np.ones(n)
    Vphi = 0.05**2*np.ones(n)
    nu_h = 3*np.ones(n)
    S_h = 0.1*(nu_h - 1)

    # construct the regressor matrix Z = [1, y_{t-1}', ..., y_{t-p}']
    tmpY = np.vstack((Y0[-p:, :], Yt))
    Z = np.zeros((Tt, n*p))
    for i in range(1, p + 1):
        Z[:, (i-1)*n:i*n] = tmpY[p-i:len(tmpY)-i, :]
    Z = np.column_stack((np.ones(Tt), Z))

    # Minnesota structure: beta0 = 0 and the constants C_ij (V_Minn with
    # unit tightness); is_own/is_cross flag own- and cross-lag positions
    beta0, V_alpha, s2_hat, _ = Minn_indep(p, kappa1, kappa2, kappa3, Y0, Yt, 0)
    _, C_Minn, _, _ = Minn_indep(p, kappa1, 1, 1, Y0, Yt, 0)
    is_int = np.zeros(n*k, dtype=bool)
    is_own = np.zeros(n*k, dtype=bool)
    for i in range(n):
        is_int[i*k] = True
        for l in range(p):
            is_own[i*k + 1 + l*n + i] = True
    is_cross = ~is_int & ~is_own
    nlag = n*k - n   # number of lag coefficients

    # prior on the free elements of B0, Minnesota-scaled:
    # B0[i, j] ~ N(0, s2_i/s2_j) for j < i
    V_b0 = np.zeros(n*(n-1)//2)
    cnt = 0
    for i in range(1, n):
        V_b0[cnt:cnt+i] = s2_hat[i]/s2_hat[:i]
        cnt += i

    # regressor row for forecasting Y[Tt, :]
    xtp1 = np.concatenate(([1.0], Yt[-1:-p-1:-1, :].ravel()))

    # initialize the Markov chain
    A = np.zeros((k, n))
    mu = np.zeros(n)
    ZZ = Z.T @ Z
    for i in range(n):
        iVai = 1/V_alpha[i*k:(i+1)*k]
        Ki = ZZ.copy()
        Ki[np.diag_indices(k)] += iVai
        A[:, i] = np.linalg.solve(Ki, Z.T @ Yt[:, i])
        mu[i] = np.log(np.mean((Yt[:, i] - Z @ A[:, i])**2))
    h = np.tile(mu, (Tt, 1))
    phi = phi0.copy()
    sigh2 = 0.05*np.ones(n)
    B0 = np.eye(n)
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
        elif prior_type == 'minnH':
            V_alpha[is_own] = kappa2*C_Minn[is_own]
            V_alpha[is_cross] = kappa3*C_Minn[is_cross]
        elif prior_type == 'mahp':
            V_alpha[is_own] = kappa2*tau2[is_own]*C_Minn[is_own]
            V_alpha[is_cross] = kappa3*tau2[is_cross]*C_Minn[is_cross]

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

        # sample the free elements of B0 row by row
        E = Yt - Z @ A
        cnt = 0
        for i in range(1, n):
            Xb = -E[:, :i]
            wb = eh_inv[:, i]
            Kb = Xb.T @ (wb[:, None]*Xb)
            Kb[np.diag_indices(i)] += 1/V_b0[cnt:cnt+i]
            CKb = np.linalg.cholesky(Kb)
            b_hat = cho_solve((CKb, True), Xb.T @ (wb*E[:, i]))
            B0[i, :i] = b_hat + solve_triangular(CKb.T, np.random.randn(i),
                                                 lower=False)
            cnt += i

        # sample the log-volatilities equation by equation
        Eorth = E @ B0.T
        ystar = np.log(Eorth**2 + 1e-4)
        for i in range(n):
            h[:, i] = SVAR1(ystar[:, i], h[:, i], mu[i], phi[i], sigh2[i])

        # sample the volatility parameters (mu, phi, sigh2)
        mu, phi, sigh2 = sample_SVAR1para_mu(h, mu, phi, sigh2, mu0, Vmu,
                                             phi0, Vphi, nu_h, S_h,
                                             np.ones(n, dtype=bool))

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
            htp1 = mu + phi*(h[-1, :] - mu) + np.sqrt(sigh2)*np.random.randn(n)
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

    # aggregate across draws
    pf = store_mu.mean(axis=0)
    tmpmax = store_lden.max(axis=0)
    lpls = np.log(np.mean(np.exp(store_lden - tmpmax), axis=0)) + tmpmax
    lpl = lpls[:q]
    lpl_joint = lpls[q]
    return pf, lpl_joint, lpl
