"""pred_VAR_homo_adapt.py
Adaptive-Minnesota version of pred_VAR_homo.py. Identical two-block
Gibbs sampler for the homoskedastic VAR, except that the Minnesota
shrinkage hyperparameters kappa2 (own-lag) and kappa3 (cross-lag) are
ESTIMATED from the data via an additional Gibbs step. Given the base
divisors d (d_own = 1/l^2, d_cross = s_i^2/(l^2 s_j^2)) so that
  Var(own a_{l,ii})    = kappa2 * d_own,
  Var(cross a_{l,ij})  = kappa3 * d_cross,
and independent IG(nu_k, S_k) priors on kappa2, kappa3 with zero prior
mean on the coefficients, the full conditionals are conjugate:
  kappa2 | beta ~ IG(nu_k2 + n_own/2,   S_k2 + 0.5*sum_own  b_j^2/d_j),
  kappa3 | beta ~ IG(nu_k3 + n_cross/2, S_k3 + 0.5*sum_cross b_j^2/d_j).
The intercept prior variance (kappa1) is held fixed.

Extra input:
  s2_hat:   length-n univariate AR(p) residual variances (from Minn_indep)
Extra outputs:
  k2m, k3m: posterior means of kappa2, kappa3
"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import invwishart


def pred_VAR_homo_adapt(Yt, Z, xt, beta0, V_Minn, s2_hat, nsim, burnin,
                        var_idx, yreal):
    T, n = Yt.shape
    k = Z.shape[1]
    p = (k - 1)//n
    V_Minn = V_Minn.copy()         # updated in place below

    # adaptive-Minnesota IG priors on kappa2, kappa3 (prior means 0.2^2, 0.2^2/4)
    nu_k2 = 3
    S_k2 = 2*0.2**2
    nu_k3 = 3
    S_k3 = 2*0.2**2/4

    # own- vs cross-lag positions (0-based) and base divisors (mirror the
    # Minn_indep index loop)
    n_own = n*p
    n_cross = n*(n-1)*p
    own_idx = np.zeros(n_own, dtype=int)
    d_own = np.zeros(n_own)
    cross_idx = np.zeros(n_cross, dtype=int)
    d_cross = np.zeros(n_cross)
    co = 0
    cc = 0
    count = 0
    for i in range(n):
        count += 1                 # intercept position (kappa1, fixed)
        for l in range(1, p + 1):
            for j in range(n):
                if i == j:
                    own_idx[co] = count
                    d_own[co] = 1/l**2
                    co += 1
                else:
                    cross_idx[cc] = count
                    d_cross[cc] = s2_hat[i]/(l**2*s2_hat[j])
                    cc += 1
                count += 1

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
    store_k2 = np.zeros(nsim)
    store_k3 = np.zeros(nsim)

    for isim in range(nsim + burnin):
        # sample beta
        Kbeta = np.diag(iVbeta) + np.kron(iSig, ZZ)
        Cbeta = np.linalg.cholesky(Kbeta)
        beta_hat = solve_triangular(Cbeta.T, solve_triangular(
            Cbeta, iVbeta*beta0 + (ZY @ iSig).flatten(order='F'),
            lower=True), lower=False)
        beta = beta_hat + solve_triangular(Cbeta.T, np.random.randn(n*k),
                                           lower=False)

        # sample kappa2, kappa3 and rebuild V_Minn / iVbeta (intercepts fixed)
        # np.random.gamma uses (shape, scale)
        b_own = beta[own_idx] - beta0[own_idx]
        b_cross = beta[cross_idx] - beta0[cross_idx]
        kappa2 = 1/np.random.gamma(nu_k2 + n_own/2,
                                   1/(S_k2 + 0.5*np.sum(b_own**2/d_own)))
        kappa3 = 1/np.random.gamma(nu_k3 + n_cross/2,
                                   1/(S_k3 + 0.5*np.sum(b_cross**2/d_cross)))
        V_Minn[own_idx] = kappa2*d_own
        V_Minn[cross_idx] = kappa3*d_cross
        iVbeta = 1/V_Minn

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
            store_k2[isave] = kappa2
            store_k3[isave] = kappa3

    # aggregate across draws
    pf = np.mean(store_mu)
    log_w = -0.5*np.log(2*np.pi*store_var) - 0.5*(yreal - store_mu)**2/store_var
    lpl = np.log(np.mean(np.exp(log_w - log_w.max()))) + log_w.max()
    k2m = np.mean(store_k2)
    k3m = np.mean(store_k3)
    return pf, lpl, k2m, k3m
