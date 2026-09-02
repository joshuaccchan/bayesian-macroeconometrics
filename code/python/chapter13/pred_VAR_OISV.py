"""pred_VAR_OISV.py
Computes the one-step-ahead posterior predictive mean
and log predictive likelihood of y_{t, var_idx} from the VAR with
order-invariant stochastic volatility,
    y_t = x_t' A  + eps_t,    eps_t ~ N(0, Sig_t),
    Sig_t^{-1} = B_0' D_t^{-1} B_0,    D_t = diag(exp(h_{1t}),...,exp(h_{nt})),
    h_{i,t} = phi_i*h_{i,t-1} + u_{i,t}^h,    u_{i,t}^h ~ N(0, sigh2_i),
    h_{i,1} ~ N(0, sigh2_i/(1 - phi_i^2)),
with the independent Minnesota prior beta ~ N(beta0, diag(V_Minn)),
N(b_{0,i}, V_b) prior on each row b_i of B_0,
TN_{(-1,1)}(phi_0, V_phi) prior on phi_i, and
IG(nu_h, S_h) prior on each sigh2_i. Five-block Gibbs sampler
following Section 13.1.3:
  1) B_0 row by row via the Waggoner-Zha-Villani approach
  2) beta given B_0 and h (Gaussian regression)
  3) h_{i,1:T} via SVAR1 (stationary AR(1) auxiliary mixture sampler)
  4) phi_i via independence-chain Metropolis-Hastings
  5) sigh2_i ~ IG

Requires: SURform2.py, sample_B0.py, SVAR1.py, sample_SVAR1para.py

Inputs:
  Yt:       T x n matrix of observations up to time t-1
  Z:        T x k matrix of regressors (intercept + p lags), k = 1+n*p
  xt:       length-k regressor row for forecasting y_t
  beta0:    length-(n*k) Minnesota prior mean of beta
  V_Minn:   length-(n*k) diagonal entries of the Minnesota prior covariance
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
from sample_B0 import sample_B0
from SVAR1 import SVAR1
from sample_SVAR1para import sample_SVAR1para


def pred_VAR_OISV(Yt, Z, xt, beta0, V_Minn, nsim, burnin, var_idx, yreal):
    T, n = Yt.shape
    k = Z.shape[1]
    iVbeta = 1/V_Minn              # diagonal prior precision (held as vector)

    # OI-SV-specific priors
    b0_B0 = np.eye(n)   # prior mean: row i is e_i
    iV_B0 = np.eye(n)   # prior precision of each row (V_b = I)
    phi_0 = 0.9
    V_phi = 0.2**2
    nu_h = 3*np.ones(n)
    S_h = 0.1*np.ones(n)

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
    phi = 0.9*np.ones(n)
    B0 = np.eye(n)

    store_mu = np.zeros(nsim)
    store_var = np.zeros(nsim)

    for isim in range(nsim + burnin):
        # sample B_0 row by row via Waggoner-Zha-Villani
        A = beta.reshape(k, n, order='F')
        E = Yt - Z @ A
        B0 = sample_B0(B0, E, h, b0_B0, iV_B0)

        # sample beta
        bigB0 = sparse.kron(sparse.eye_array(T), sparse.csc_array(B0),
                            format='csc')
        iD = sparse.diags_array(np.exp(-h).flatten(), format='csc')
        iSig = bigB0.T @ iD @ bigB0
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
        Eorth = E @ B0.T
        ystar = np.log(Eorth**2 + 1e-4)
        for i in range(n):
            h[:, i] = SVAR1(ystar[:, i], h[:, i], 0, phi[i], sigh2[i])

        # sample phi and sigh2
        phi, sigh2 = sample_SVAR1para(h, phi, sigh2, phi_0, V_phi, nu_h, S_h)

        if isim >= burnin:
            isave = isim - burnin
            mu_full = xt @ A
            # forecast h_{T+1} via stationary AR(1)
            h_tp1 = phi*h[-1, :] + np.sqrt(sigh2)*np.random.randn(n)
            iB0 = np.linalg.inv(B0)
            Sig_tp1 = iB0 @ np.diag(np.exp(h_tp1)) @ iB0.T
            store_mu[isave] = mu_full[var_idx]
            store_var[isave] = Sig_tp1[var_idx, var_idx]

    # aggregate across draws
    pf = np.mean(store_mu)
    log_w = -0.5*np.log(2*np.pi*store_var) \
        - 0.5*(yreal - store_mu)**2/store_var
    lpl = np.log(np.mean(np.exp(log_w - log_w.max()))) + log_w.max()
    return pf, lpl
