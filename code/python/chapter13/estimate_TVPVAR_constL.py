"""estimate_TVPVAR_constL.py
Gibbs sampler for the TVP-VAR with stochastic volatility in which the
lower-triangular factor L in Sigma_t^{-1} = L' * D_t^{-1} * L is held
CONSTANT over time (L_t = L for all t). This is the Cogley-Sargent (2005)
specification: only the VAR coefficients beta_t and the log-volatilities
h_t are time-varying, while the contemporaneous relations L are constant.

Model
  y_t = X_t * beta_t + eps_t,   eps_t ~ N(0, Sigma_t),
  Sigma_t^{-1} = L' * D_t^{-1} * L,     (L constant, unit lower triangular)
  D_t = diag(exp(h_{1t}), ..., exp(h_{nt})),
where l collects the m = n(n-1)/2 free elements of L stacked
equation by equation  l = (l_{2,1}, l_{3,1}, l_{3,2}, ..., l_{n,n-1})'.
State equations
  beta_t = beta_{t-1} + u_t^beta,   u_t^beta ~ N(0, Q),
  h_{it} = h_{i,t-1}  + u_{it}^h,   u_{it}^h ~ N(0, sigh2_i),
with full Q and Gaussian priors on (beta_0, h_0). The constant l has a
fixed Gaussian prior l ~ N(m_l, V_l).

Relative to estimate_TVPVAR.py, Step 2 (time-varying l path with random-walk
innovation covariance S) is replaced by a single constant-l Gaussian draw,
and the S block / l_0 initial condition are removed. Blocks 1 (beta RW),
3 (h RW SV), 4 (beta_0, h_0), and the Q, sigh2 hyperparameter draws are
unchanged from estimate_TVPVAR.py.

Requires: SURform.py, SVRW.py

Inputs
  Y      : (T_all) x n data, with the FIRST p ROWS used only as
           initial lags. Effective sample length is T = T_all - p.
  p      : number of VAR lags.
  prior  : dict with keys
      beta0_mean (nk,), beta0_var  (nk x nk SPD)
      l0_mean    (m,),  l0_var     (m  x m  SPD)   # used as (m_l, V_l)
      h0_mean    (n,),  h0_var     (n  x n  SPD)
      Q_nu (scalar), Q_S0 (nk x nk SPD; full IW prior scale)
      sigh2_nu (n,), sigh2_S0 (n,)
    (Keys S_nu, S_S0 are ignored if present.)
  nsim, burnin : MCMC settings.

Outputs (post-burn-in)
  store_beta  : nsim x (T*nk)
  store_l     : nsim x m           (constant l for each draw)
  store_h     : nsim x T x n
  store_Sig_t : empty (residual std devs recomputed from store_l, store_h)
  store_Q     : nsim x (nk*nk)
  store_sigh2 : nsim x n
  store_beta0 : nsim x nk
  store_h0    : nsim x n
"""

import numpy as np
from scipy import sparse
from scipy.linalg import (cholesky_banded, solveh_banded, solve_banded,
                          solve_triangular)
from scipy.stats import invwishart

from SURform import SURform
from SVRW import SVRW


def estimate_TVPVAR_constL(Y, p, prior, nsim, burnin):
    Tall, n = Y.shape
    k = 1 + n*p
    nk = n*k
    m = n*(n - 1)//2
    T = Tall - p

    # construct X and y
    Yeff = Y[p:, :]
    X_tilde = np.zeros((T, n*p))
    for i in range(1, p + 1):
        X_tilde[:, (i-1)*n:i*n] = Y[p-i:Tall-i, :]
    y = Yeff.flatten()                     # reshape(Yeff', n*T, 1)
    X = SURform(np.hstack((np.ones((n*T, 1)),
                           np.kron(X_tilde, np.ones((n, 1))))))

    # T x T first-difference operator H and its Gram matrix HtH = H'*H
    H = sparse.eye_array(T, format='csc') - sparse.csc_array(
        (np.ones(T-1), (np.arange(1, T), np.arange(T-1))), shape=(T, T))
    HtH = (H.T @ H).tocsc()

    # prior
    beta0_mean = np.asarray(prior['beta0_mean']).flatten()
    iVbeta0 = np.linalg.inv(prior['beta0_var'])
    l_mean = np.asarray(prior['l0_mean']).flatten()  # prior mean m_l of constant l
    iVl = np.linalg.inv(prior['l0_var'])             # prior precision V_l^{-1}
    h0_mean = np.asarray(prior['h0_mean']).flatten()
    iVh0 = np.linalg.inv(prior['h0_var'])
    Q_nu = prior['Q_nu']
    Q_S0 = prior['Q_S0']
    sigh2_nu = np.asarray(prior['sigh2_nu']).flatten()
    sigh2_S0 = np.asarray(prior['sigh2_S0']).flatten()

    # 0-based (row, col) positions of the free entries below the diagonal of
    # L, stacked equation by equation: (1,0), (2,0), (2,1), (3,0), ...
    r_pos = np.array([i for i in range(1, n) for _ in range(i)])
    c_pos = np.array([j for i in range(1, n) for j in range(i)])

    # initialize chain at zero states (h = 0 implies sigma = 1) with tiny
    # innovation covariances so the first beta draw is heavily smoothed.
    l = l_mean.copy()          # constant l initialized at prior mean
    h = np.zeros((T, n))
    Q = 1e-4*np.eye(nk)
    sigh2 = 1e-4*np.ones(n)

    # initial conditions
    beta0 = beta0_mean.copy()
    h0 = h0_mean.copy()

    # storage
    store_beta = np.zeros((nsim, T*nk))
    store_l = np.zeros((nsim, m))
    store_h = np.zeros((nsim, T, n))
    store_Sig_t = np.empty(0)
    store_Q = np.zeros((nsim, nk*nk))
    store_sigh2 = np.zeros((nsim, n))
    store_beta0 = np.zeros((nsim, nk))
    store_h0 = np.zeros((nsim, n))

    # Kbeta is block tridiagonal with block size nk, hence banded
    u_b = 2*nk - 1

    # MCMC starts here
    for isim in range(1, nsim + burnin + 1):
        # replicate the constant l across all T blocks for the sparse builders
        l_rep = np.tile(l, T)

        # ---- Block 1: sample beta (unchanged) ----
        iSig_big = build_iSig(l_rep, h, n, T, r_pos, c_pos)
        iQ = np.linalg.inv(Q)
        HiQH = sparse.kron(HtH, sparse.csc_array(iQ), format='csc')
        rhs_b = np.zeros(T*nk)
        rhs_b[:nk] = iQ @ beta0
        XiSig = (X.T @ iSig_big).tocsc()
        Kbeta = (HiQH + XiSig @ X).tocsc()
        ab = band_from_sparse(Kbeta, u_b)
        beta_hat = solveh_banded(ab, rhs_b + XiSig @ y)
        Ub = cholesky_banded(ab)           # Kbeta = Ub'Ub with Ub upper banded
        beta = beta_hat + solve_banded((0, u_b), Ub, np.random.randn(T*nk))

        # ---- Block 2: sample CONSTANT l ----
        # eps_t = E_t l + u_t, u_t ~ N(0, D_t), stacked over all t:
        #   (l | .) ~ N(lhat, Kl^{-1}),  Kl = V_l^{-1} + E' D^{-1} E,
        #   lhat = Kl^{-1} (V_l^{-1} m_l + E' D^{-1} eps).
        eps_full = y - X @ beta
        U = eps_full.reshape(T, n)         # reshape(eps_full, n, T)'
        E_big = build_E_const(U, n, T, r_pos, c_pos)
        iD = sparse.diags_array(np.exp(-h.flatten()), format='csc')
        EiD = E_big.T @ iD
        Kl = iVl + (EiD @ E_big).toarray()   # m x m, small dense
        Cl = np.linalg.cholesky(Kl)
        l_hat = solve_triangular(Cl.T, solve_triangular(
            Cl, iVl @ l_mean + EiD @ eps_full, lower=True), lower=False)
        l = l_hat + solve_triangular(Cl.T, np.random.randn(m), lower=False)

        # ---- Block 3: sample h equation by equation (unchanged) ----
        l_rep = np.tile(l, T)
        Eorth = (build_L(l_rep, n, T, r_pos, c_pos) @ eps_full).reshape(T, n)
        ystar = np.log(Eorth**2 + 1e-4)
        for ii in range(n):
            h[:, ii] = SVRW(ystar[:, ii], h[:, ii], h0[ii], sigh2[ii])

        # ---- Block 4: sample beta0, h0 (l0 removed) ----
        Kb0 = iVbeta0 + iQ
        Cb0 = np.linalg.cholesky(Kb0)
        b0_hat = solve_triangular(Cb0.T, solve_triangular(
            Cb0, iVbeta0 @ beta0_mean + iQ @ beta[:nk], lower=True),
            lower=False)
        beta0 = b0_hat + solve_triangular(Cb0.T, np.random.randn(nk),
                                          lower=False)

        Kh0 = iVh0 + np.diag(1/sigh2)
        Ch0 = np.linalg.cholesky(Kh0)
        h0_hat = solve_triangular(Ch0.T, solve_triangular(
            Ch0, iVh0 @ h0_mean + h[0, :]/sigh2, lower=True), lower=False)
        h0 = h0_hat + solve_triangular(Ch0.T, np.random.randn(n), lower=False)

        # ---- Block 5: sample Q, sigh2 (S removed) ----
        bmat = beta.reshape(T, nk).T       # reshape(beta, nk, T)
        db = bmat - np.column_stack((beta0, bmat[:, :-1]))
        Q = invwishart.rvs(df=Q_nu + T, scale=Q_S0 + db @ db.T)

        dh = h - np.vstack((h0, h[:-1, :]))
        # np.random.gamma uses (shape, scale)
        sigh2 = 1/np.random.gamma(sigh2_nu + T/2,
                                  1/(sigh2_S0 + 0.5*np.sum(dh**2, axis=0)))

        if isim > burnin:
            isave = isim - burnin - 1
            store_beta[isave, :] = beta
            store_l[isave, :] = l
            store_h[isave, :, :] = h
            store_Q[isave, :] = Q.flatten(order='F')
            store_sigh2[isave, :] = sigh2
            store_beta0[isave, :] = beta0
            store_h0[isave, :] = h0

        if isim % 1000 == 0:
            print(f'estimate_TVPVAR_constL: iteration {isim} / '
                  f'{nsim + burnin}', flush=True)

    return (store_beta, store_l, store_h, store_Sig_t, store_Q,
            store_sigh2, store_beta0, store_h0)


# =====================================================================
def band_from_sparse(K, u):
    """Upper banded storage of the sparse symmetric matrix K with u
    superdiagonals: row u-i holds the i-th superdiagonal (the scipy
    *_banded convention)."""
    N = K.shape[0]
    ab = np.zeros((u + 1, N))
    for i in range(u + 1):
        ab[u - i, i:] = K.diagonal(i)
    return ab


# =====================================================================
def build_iSig(l, h, n, T, r_pos, c_pos):
    """Assemble sparse blkdiag(L' D_t^{-1} L, t=1..T), size (n*T) x (n*T)."""
    L_big = build_L(l, n, T, r_pos, c_pos)
    iD = sparse.diags_array(np.exp(-h.flatten()), format='csc')
    return (L_big.T @ iD @ L_big).tocsc()


# =====================================================================
def build_L(l, n, T, r_pos, c_pos):
    """Assemble sparse L_big = blkdiag(L_1, ..., L_T), size (n*T) x (n*T),
    where each L_t is unit lower-triangular with free entries given by
    the t-th block of length m of the stacked vector l."""
    m = n*(n - 1)//2

    t_off_n = np.arange(T)[:, None]*n     # T x 1
    I = t_off_n + r_pos                   # T x m
    J = t_off_n + c_pos                   # T x m
    V = l.reshape(T, m)                   # row t is l_t'

    diag_idx = np.arange(n*T)
    return sparse.csc_array(
        (np.concatenate((np.ones(n*T), V.ravel())),
         (np.concatenate((diag_idx, I.ravel())),
          np.concatenate((diag_idx, J.ravel())))), shape=(n*T, n*T))


# =====================================================================
def build_E_const(U, n, T, r_pos, c_pos):
    """Assemble sparse E (n*T x m) mapping the CONSTANT l to the stacked
    n*T vector with block E_t in rows t*n : (t+1)*n. All blocks share
    the SAME m columns (unlike the time-varying case), so E_t l = L*eps_t
    - eps_t. Row r_pos, column pos carries -eps_{c_pos(pos), t}."""
    m = n*(n - 1)//2

    t_off_n = np.arange(T)[:, None]*n     # T x 1
    I = t_off_n + r_pos                   # T x m
    J = np.tile(np.arange(m), (T, 1))     # T x m; columns fixed (no t offset)
    V = -U[:, c_pos]                      # T x m

    return sparse.csc_array((V.ravel(), (I.ravel(), J.ravel())),
                            shape=(n*T, m))
