"""pred_VAR_homo.py
Computes the one-step-ahead posterior predictive mean
and log predictive likelihood of y_{t, var_idx} from the
homoskedastic VAR
    y_t = A' x_t + eps_t,   eps_t ~ N(0, Sig),
with the independent Minnesota prior beta ~ N(beta0, diag(V_Minn))
and the inverse-Wishart prior Sig ~ IW(nu0, S0). Two-block Gibbs
sampler. The point forecast and LPL are aggregated within the
function as the posterior predictive mean and the log-mean-exp of
the per-draw Gaussian densities.

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
from scipy.linalg import solve_triangular
from scipy.stats import invwishart


def pred_VAR_homo(Yt, Z, xt, beta0, V_Minn, nsim, burnin, var_idx, yreal):
    T, n = Yt.shape
    k = Z.shape[1]
    iVbeta = 1/V_Minn              # diagonal prior precision (held as vector)

    # inverse-Wishart prior on Sig
    nu0 = n + 2
    S0 = np.eye(n)

    # precompute
    ZZ = Z.T @ Z
    ZY = Z.T @ Yt

    # initialize the Gibbs sampler at OLS
    A = np.linalg.lstsq(Z, Yt, rcond=None)[0]
    beta = A.flatten(order='F')
    E = Yt - Z @ A
    Sig = E.T @ E/T
    iSig = np.linalg.inv(Sig)

    store_mu = np.zeros(nsim)
    store_var = np.zeros(nsim)

    for isim in range(nsim + burnin):
        # sample beta
        Kbeta = np.diag(iVbeta) + np.kron(iSig, ZZ)
        Cbeta = np.linalg.cholesky(Kbeta)
        beta_hat = solve_triangular(Cbeta.T, solve_triangular(
            Cbeta, iVbeta*beta0 + (ZY @ iSig).flatten(order='F'),
            lower=True), lower=False)
        beta = beta_hat + solve_triangular(Cbeta.T, np.random.randn(n*k),
                                           lower=False)

        # sample Sig
        A = beta.reshape(k, n, order='F')
        E = Yt - Z @ A
        Sig = invwishart.rvs(df=nu0 + T, scale=S0 + E.T @ E)
        iSig = np.linalg.inv(Sig)

        if isim >= burnin:
            isave = isim - burnin
            mu_full = xt @ A
            store_mu[isave] = mu_full[var_idx]
            store_var[isave] = Sig[var_idx, var_idx]

    # aggregate across draws
    pf = np.mean(store_mu)
    log_w = -0.5*np.log(2*np.pi*store_var) - 0.5*(yreal - store_mu)**2/store_var
    lpl = np.log(np.mean(np.exp(log_w - log_w.max()))) + log_w.max()
    return pf, lpl
