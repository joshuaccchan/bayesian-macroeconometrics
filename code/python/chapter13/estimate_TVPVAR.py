"""estimate_TVPVAR.py
Implements a Gibbs sampler for the TVP-VAR with stochastic
volatility of Primiceri (2005); see Section 13.2 of the textbook.

Model
  y_t = X_t * beta_t + eps_t,   eps_t ~ N(0, Sigma_t),
  Sigma_t^{-1} = L_t' * D_t^{-1} * L_t,
  D_t = diag(exp(h_{1t}), ..., exp(h_{nt})),
where L_t is unit-lower-triangular and l_t collects its m=n(n-1)/2
free elements stacked equation by equation
    l_t = (l_{2,1,t}, l_{3,1,t}, l_{3,2,t}, ..., l_{n,n-1,t})'.
State equations
  beta_t = beta_{t-1} + u_t^beta,    u_t^beta ~ N(0, Q),
  l_t    = l_{t-1}    + u_t^l,       u_t^l    ~ N(0, S),
  h_{it} = h_{i,t-1}  + u_{it}^h,    u_{it}^h ~ N(0, sigh2_i),
with full Q, block-diagonal S = diag(S_1,...,S_{n-1}), and
Gaussian priors on (beta_0, l_0, h_0).

Requires: SURform.py, SVRW.py

Inputs
  Y      : (T_all) x n data, with the FIRST p ROWS used only as
           initial lags. Effective sample length is T = T_all - p.
  p      : number of VAR lags.
  prior  : dict with keys
      beta0_mean (nk,), beta0_var  (nk x nk SPD)
      l0_mean    (m,),  l0_var     (m  x m  SPD)
      h0_mean    (n,),  h0_var     (n  x n  SPD)
      Q_nu (scalar), Q_S0 (nk x nk SPD; full IW prior scale)
      S_nu (length n-1), S_S0 (length n-1 list, S_S0[j-1] is j x j SPD)
      sigh2_nu (n,), sigh2_S0 (n,)
  nsim, burnin : MCMC settings.

Outputs (post-burn-in)
  store_beta  : nsim x (T*nk)
  store_l     : nsim x (T*m)
  store_h     : nsim x T x n
  store_Sig_t : nsim x (n*n) x T   (empty if too large)
  store_Q     : nsim x (nk*nk)
  store_S     : nsim x (m*m)
  store_sigh2 : nsim x n
  store_beta0 : nsim x nk
  store_l0    : nsim x m
  store_h0    : nsim x n
"""

import numpy as np
from scipy import sparse
from scipy.linalg import (block_diag, cholesky_banded, solveh_banded,
                          solve_banded, solve_triangular)
from scipy.stats import invwishart

from SURform import SURform
from SVRW import SVRW


def estimate_TVPVAR(Y, p, prior, nsim, burnin):
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
    # The random-walk precision on a stacked (T*r)-vector of states with inner
    # covariance A is kron(HtH, inv(A)).
    H = sparse.eye_array(T, format='csc') - sparse.csc_array(
        (np.ones(T-1), (np.arange(1, T), np.arange(T-1))), shape=(T, T))
    HtH = (H.T @ H).tocsc()

    # prior
    beta0_mean = np.asarray(prior['beta0_mean']).flatten()
    iVbeta0 = np.linalg.inv(prior['beta0_var'])
    l0_mean = np.asarray(prior['l0_mean']).flatten()
    iVl0 = np.linalg.inv(prior['l0_var'])
    h0_mean = np.asarray(prior['h0_mean']).flatten()
    iVh0 = np.linalg.inv(prior['h0_var'])
    Q_nu = prior['Q_nu']
    Q_S0 = prior['Q_S0']
    S_nu = np.asarray(prior['S_nu'])
    S_S0 = prior['S_S0']
    sigh2_nu = np.asarray(prior['sigh2_nu']).flatten()
    sigh2_S0 = np.asarray(prior['sigh2_S0']).flatten()

    # 0-based (row, col) positions of the free entries below the diagonal of
    # L, stacked equation by equation: (1,0), (2,0), (2,1), (3,0), ...
    # (replaces the MATLAB linear-index vector L_id)
    r_pos = np.array([i for i in range(1, n) for _ in range(i)])
    c_pos = np.array([j for i in range(1, n) for j in range(i)])

    # initialize chain at zero states (h = 0 implies sigma = 1) with tiny
    # innovation covariances so the first beta draw is heavily smoothed.
    l = np.zeros(T*m)
    h = np.zeros((T, n))
    Q = 1e-4*np.eye(nk)
    sigh2 = 1e-4*np.ones(n)
    S = [1e-4*np.eye(ii) for ii in range(1, n)]

    # initial conditions
    beta0 = beta0_mean.copy()
    l0 = l0_mean.copy()
    h0 = h0_mean.copy()

    # storage
    store_beta = np.zeros((nsim, T*nk))
    store_l = np.zeros((nsim, T*m))
    store_h = np.zeros((nsim, T, n))
    populate_Sig = (n*n*T*nsim < 5e7)
    if populate_Sig:
        store_Sig_t = np.zeros((nsim, n*n, T))
    else:
        store_Sig_t = np.empty(0)
    store_Q = np.zeros((nsim, nk*nk))
    store_S = np.zeros((nsim, m*m))
    store_sigh2 = np.zeros((nsim, n))
    store_beta0 = np.zeros((nsim, nk))
    store_l0 = np.zeros((nsim, m))
    store_h0 = np.zeros((nsim, n))

    # the stacked-state precision matrices are block tridiagonal (block sizes
    # nk and m), hence banded; they are factored in banded storage throughout
    u_b = 2*nk - 1                     # bandwidth of Kbeta
    u_l = 2*m - 1                      # bandwidth of Kl
    diag_n = np.arange(n)

    # MCMC starts here
    for isim in range(1, nsim + burnin + 1):
        # sample beta
        iSig_big = build_iSig(l, h, n, T, r_pos, c_pos)
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

        # sample l
        eps_full = y - X @ beta
        U = eps_full.reshape(T, n)         # reshape(eps_full, n, T)'
        E_big = build_E(U, n, T, r_pos, c_pos)
        iD = sparse.diags_array(np.exp(-h.flatten()), format='csc')
        # Block-diagonal S^{-1} (m x m), with blocks S_i^{-1}
        iSmat = blockdiag_inv(S)
        HliSHl = sparse.kron(HtH, sparse.csc_array(iSmat), format='csc')
        rhs_l = np.zeros(T*m)
        rhs_l[:m] = iSmat @ l0
        EiD = (E_big.T @ iD).tocsc()
        Kl = (HliSHl + EiD @ E_big).tocsc()
        abl = band_from_sparse(Kl, u_l)
        l_hat = solveh_banded(abl, rhs_l + EiD @ eps_full)
        Ul = cholesky_banded(abl)          # Kl = Ul'Ul with Ul upper banded
        l = l_hat + solve_banded((0, u_l), Ul, np.random.randn(T*m))

        # sample h equation by equation (eps_full unchanged since beta fixed)
        Eorth = (build_L(l, n, T, r_pos, c_pos) @ eps_full).reshape(T, n)
        ystar = np.log(Eorth**2 + 1e-4)
        for ii in range(n):
            h[:, ii] = SVRW(ystar[:, ii], h[:, ii], h0[ii], sigh2[ii])

        # sample beta0, l0, h0 (with diffuse priors); reuse iQ and iSmat
        Kb0 = iVbeta0 + iQ
        Cb0 = np.linalg.cholesky(Kb0)
        b0_hat = solve_triangular(Cb0.T, solve_triangular(
            Cb0, iVbeta0 @ beta0_mean + iQ @ beta[:nk], lower=True),
            lower=False)
        beta0 = b0_hat + solve_triangular(Cb0.T, np.random.randn(nk),
                                          lower=False)

        Kl0 = iVl0 + iSmat
        Cl0 = np.linalg.cholesky(Kl0)
        l0_hat = solve_triangular(Cl0.T, solve_triangular(
            Cl0, iVl0 @ l0_mean + iSmat @ l[:m], lower=True), lower=False)
        l0 = l0_hat + solve_triangular(Cl0.T, np.random.randn(m), lower=False)

        Kh0 = iVh0 + np.diag(1/sigh2)
        Ch0 = np.linalg.cholesky(Kh0)
        h0_hat = solve_triangular(Ch0.T, solve_triangular(
            Ch0, iVh0 @ h0_mean + h[0, :]/sigh2, lower=True), lower=False)
        h0 = h0_hat + solve_triangular(Ch0.T, np.random.randn(n), lower=False)

        # sample Q, S, sigh2
        bmat = beta.reshape(T, nk).T       # reshape(beta, nk, T)
        db = bmat - np.column_stack((beta0, bmat[:, :-1]))
        Q = invwishart.rvs(df=Q_nu + T, scale=Q_S0 + db @ db.T)
        lmat = l.reshape(T, m).T
        dl = lmat - np.column_stack((l0, lmat[:, :-1]))
        off = 0
        for ii in range(1, n):
            di = ii
            DLi = dl[off:off+di, :]
            S[ii-1] = np.atleast_2d(invwishart.rvs(
                df=S_nu[ii-1] + T, scale=S_S0[ii-1] + DLi @ DLi.T))
            off += di

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
            Sfull = np.zeros((m, m))
            off = 0
            for ii in range(1, n):
                di = ii
                Sfull[off:off+di, off:off+di] = S[ii-1]
                off += di
            store_S[isave, :] = Sfull.flatten(order='F')
            store_sigh2[isave, :] = sigh2
            store_beta0[isave, :] = beta0
            store_l0[isave, :] = l0
            store_h0[isave, :] = h0
            if populate_Sig:
                # Sigma_t = L_t^{-1} D_t L_t^{-T} for every t via batched
                # small solves (replaces the big sparse solve in MATLAB)
                Lts = np.tile(np.eye(n), (T, 1, 1))
                Lts[:, r_pos, c_pos] = l.reshape(T, m)
                Dsqrt = np.zeros((T, n, n))
                Dsqrt[:, diag_n, diag_n] = np.exp(h/2)
                M = np.linalg.solve(Lts, Dsqrt)      # M_t = L_t \ D_t^{1/2}
                Sig_all = M @ np.transpose(M, (0, 2, 1))
                store_Sig_t[isave, :, :] = Sig_all.reshape(T, n*n).T

        if isim % 1000 == 0:
            print(f'estimate_TVPVAR: iteration {isim} / {nsim + burnin}',
                  flush=True)

    return (store_beta, store_l, store_h, store_Sig_t, store_Q, store_S,
            store_sigh2, store_beta0, store_l0, store_h0)


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
    """Assemble sparse blkdiag(L_t' D_t^{-1} L_t, t=1..T), size (n*T) x (n*T)."""
    L_big = build_L(l, n, T, r_pos, c_pos)
    iD = sparse.diags_array(np.exp(-h.flatten()), format='csc')
    return (L_big.T @ iD @ L_big).tocsc()


# =====================================================================
def build_L(l, n, T, r_pos, c_pos):
    """Assemble sparse L_big = blkdiag(L_1, ..., L_T), size (n*T) x (n*T),
    where each L_t is unit lower-triangular with free entries given by
    l_t (the t-th block of length m of the stacked vector l)."""
    m = n*(n - 1)//2

    # Off-diagonal triplets of L_big = blkdiag(L_1, ..., L_T)
    t_off_n = np.arange(T)[:, None]*n     # T x 1
    I = t_off_n + r_pos                   # T x m
    J = t_off_n + c_pos                   # T x m
    V = l.reshape(T, m)                   # row t is l_t'

    # Sparse L_big: unit diagonal in each block plus the off-diagonal entries
    diag_idx = np.arange(n*T)
    return sparse.csc_array(
        (np.concatenate((np.ones(n*T), V.ravel())),
         (np.concatenate((diag_idx, I.ravel())),
          np.concatenate((diag_idx, J.ravel())))), shape=(n*T, n*T))


# =====================================================================
def build_E(U, n, T, r_pos, c_pos):
    """Assemble sparse E (n*T x m*T) block-diagonal with the n x m block
    E_t in slot t. E_t maps l_t to L_t * eps_t - eps_t, with row r_pos
    and column pos carrying the entry -eps_{c_pos(pos), t}."""
    m = n*(n - 1)//2

    # Triplets of E_big = blkdiag(E_1, ..., E_T)
    t_off_n = np.arange(T)[:, None]*n     # T x 1
    t_off_m = np.arange(T)[:, None]*m     # T x 1
    I = t_off_n + r_pos                   # T x m
    J = t_off_m + np.arange(m)            # T x m
    V = -U[:, c_pos]                      # T x m

    return sparse.csc_array((V.ravel(), (I.ravel(), J.ravel())),
                            shape=(n*T, m*T))


# =====================================================================
def blockdiag_inv(S):
    """Dense m x m block-diagonal matrix with blocks S[i]^{-1} (m is small;
    replaces the MATLAB sparse triplet builder blockdiag_inv_triplets)."""
    return block_diag(*[np.linalg.inv(Si) for Si in S])
