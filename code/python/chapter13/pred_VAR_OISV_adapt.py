"""pred_VAR_OISV_adapt.py
Adaptive-Minnesota version of pred_VAR_OISV.py (order-invariant
stochastic volatility). Identical five-block Gibbs sampler, plus an
extra Gibbs step that estimates the Minnesota shrinkage
hyperparameters kappa2 (own-lag) and kappa3 (cross-lag) from their
conjugate IG full conditionals; the intercept prior variance (kappa1)
is held fixed.

Requires: SURform2.py, sample_B0.py, SVAR1.py, sample_SVAR1para.py

Extra input:
  s2_hat:   length-n univariate AR(p) residual variances (from Minn_indep)
Extra outputs:
  k2m, k3m: posterior means of kappa2, kappa3
"""

import numpy as np
from scipy import sparse
from scipy.linalg import solve_triangular

from SURform2 import SURform2
from sample_B0 import sample_B0
from SVAR1 import SVAR1
from sample_SVAR1para import sample_SVAR1para


def pred_VAR_OISV_adapt(Yt, Z, xt, beta0, V_Minn, s2_hat, nsim, burnin,
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
    store_k2 = np.zeros(nsim)
    store_k3 = np.zeros(nsim)

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
            store_k2[isave] = kappa2
            store_k3[isave] = kappa3

    # aggregate across draws
    pf = np.mean(store_mu)
    log_w = -0.5*np.log(2*np.pi*store_var) \
        - 0.5*(yreal - store_mu)**2/store_var
    lpl = np.log(np.mean(np.exp(log_w - log_w.max()))) + log_w.max()
    k2m = np.mean(store_k2)
    k3m = np.mean(store_k3)
    return pf, lpl, k2m, k3m
