"""pred_VAR_SV.py
Computes the one-step-ahead posterior predictive mean
and log predictive likelihood of y_{t, var_idx} from the VAR with
Cholesky stochastic volatility,
    y_t = x_t' A  + eps_t,    eps_t ~ N(0, Sig_t),
    Sig_t^{-1} = L' D_t^{-1} L,    D_t = diag(exp(h_{1t}),...,exp(h_{nt})),
    h_{i,t} = h_{i,t-1} + u_{i,t}^h,    u_{i,t}^h ~ N(0, sigh2_i),
with the independent Minnesota prior beta ~ N(beta0, diag(V_Minn)),
N(l0, V_l) prior on the free elements l of L,
IG(nu_h, S_h) prior on each sigh2_i, and
N(m_h0, V_h0) prior on h_0 = (h_{1,0},...,h_{n,0})'. Five-block Gibbs
sampler following Section 13.1.2.

Requires: SURform2.py, SVRW.py

Inputs:
  Yt:       T x n matrix of observations up to time t-1
  Z:        T x k matrix of regressors (intercept + p lags), k = 1+n*p
  xt:       length-k regressor row for forecasting y_t
  beta0:    length-nk Minnesota prior mean of beta
  V_Minn:   length-nk diagonal entries of the Minnesota prior covariance
  nsim:     scalar; number of post-burnin MCMC draws
  burnin:   scalar; number of burnin draws
  var_idx:  scalar; 0-based index of the variable to forecast
  yreal:    scalar; realized value of y_{t, var_idx} (for LPL)

Outputs:
  pf:   posterior predictive mean of y_{t, var_idx}
  lpl:  log of the predictive density at yreal, computed as the
        log-mean-exp over MCMC draws of mixture-of-Gaussians weights
"""

import numpy as np
from scipy import sparse
from scipy.linalg import solve_triangular

from SURform2 import SURform2
from SVRW import SVRW


def pred_VAR_SV(Yt, Z, xt, beta0, V_Minn, nsim, burnin, var_idx, yreal):
    T, n = Yt.shape
    k = Z.shape[1]
    m = n*(n - 1)//2
    iVbeta = 1/V_Minn              # diagonal prior precision (held as vector)

    # SV-specific priors
    l0 = np.zeros(m)
    iVl = np.eye(m)
    m_h0 = np.zeros(n)
    iVh0 = np.eye(n)/10
    nu_h = 3*np.ones(n)
    S_h = 0.1*np.ones(n)

    # 0-based (row, col) indices of the strictly lower-triangular elements
    # of L in row-major order, so that L[L_id_r, L_id_c] = l_vec corresponds to
    #   l_vec = (l_{21}, l_{31}, l_{32}, l_{41}, l_{42}, l_{43}, ...)'
    L_id_r = np.zeros(m, dtype=int)
    L_id_c = np.zeros(m, dtype=int)
    ii = 0
    for i in range(1, n):
        for j in range(i):
            L_id_r[ii] = i
            L_id_c[ii] = j
            ii += 1

    # SUR-form regressors and time-stacked observations
    X = SURform2(Z, n)
    y = Yt.flatten()               # reshape(Yt', n*T, 1)

    # initialize the Gibbs sampler at OLS
    A = np.linalg.lstsq(Z, Yt, rcond=None)[0]
    beta = A.flatten(order='F')
    E = Yt - Z @ A
    h0 = np.log(np.diag(E.T @ E/T))
    h = np.tile(h0, (T, 1))
    sigh2 = 0.1*np.ones(n)
    l_vec = np.zeros(m)
    L = np.eye(n)

    store_mu = np.zeros(nsim)
    store_var = np.zeros(nsim)

    for isim in range(nsim + burnin):
        # sample beta
        L[L_id_r, L_id_c] = l_vec
        bigL = sparse.kron(sparse.eye_array(T), sparse.csc_array(L),
                           format='csc')
        iD = sparse.diags_array(np.exp(-h).flatten(), format='csc')
        iSig = bigL.T @ iD @ bigL
        XiSig = X.T @ iSig
        Kbeta = np.diag(iVbeta) + (XiSig @ X).toarray()
        Cbeta = np.linalg.cholesky(Kbeta)
        beta_hat = solve_triangular(Cbeta.T, solve_triangular(
            Cbeta, iVbeta*beta0 + XiSig @ y, lower=True), lower=False)
        beta = beta_hat + solve_triangular(Cbeta.T, np.random.randn(n*k),
                                           lower=False)

        # sample h equation by equation
        A = beta.reshape(k, n, order='F')
        E = Yt - Z @ A
        Eorth = E @ L.T
        ystar = np.log(Eorth**2 + 1e-4)
        for i in range(n):
            h[:, i] = SVRW(ystar[:, i], h[:, i], h0[i], sigh2[i])

        # sample l_vec from the regression eps_t = E_t * l + eta_t
        Em = np.zeros((T*n, m))
        cE = 0
        for ii in range(1, n):
            Em[ii::n, cE:cE+ii] = -E[:, :ii]
            cE += ii
        w = np.exp(-h).flatten()   # diagonal of D^{-1}, time-stacked
        Kl = iVl + Em.T @ (Em*w[:, None])
        Cl = np.linalg.cholesky(Kl)
        l_hat = solve_triangular(Cl.T, solve_triangular(
            Cl, iVl @ l0 + Em.T @ (w*E.flatten()), lower=True), lower=False)
        l_vec = l_hat + solve_triangular(Cl.T, np.random.randn(m), lower=False)

        # sample sigh2; np.random.gamma uses (shape, scale)
        e2 = (h - np.vstack((h0, h[:-1, :])))**2
        sigh2 = 1/np.random.gamma(nu_h + T/2, 1/(S_h + e2.sum(axis=0)/2))

        # sample h0
        Kh0 = iVh0 + np.diag(1/sigh2)
        Ch0 = np.linalg.cholesky(Kh0)
        h0_hat = solve_triangular(Ch0.T, solve_triangular(
            Ch0, iVh0 @ m_h0 + h[0, :]/sigh2, lower=True), lower=False)
        h0 = h0_hat + solve_triangular(Ch0.T, np.random.randn(n), lower=False)

        if isim >= burnin:
            isave = isim - burnin
            mu_full = xt @ A
            # forecast h_{T+1} via random walk and the implied Sig_{T+1}
            h_tp1 = h[-1, :] + np.sqrt(sigh2)*np.random.randn(n)
            L[L_id_r, L_id_c] = l_vec
            invL = solve_triangular(L, np.eye(n), lower=True)
            Sig_tp1 = invL @ np.diag(np.exp(h_tp1)) @ invL.T
            store_mu[isave] = mu_full[var_idx]
            store_var[isave] = Sig_tp1[var_idx, var_idx]

    # aggregate across draws
    pf = np.mean(store_mu)
    log_w = -0.5*np.log(2*np.pi*store_var) - 0.5*(yreal - store_mu)**2/store_var
    lpl = np.log(np.mean(np.exp(log_w - log_w.max()))) + log_w.max()
    return pf, lpl
