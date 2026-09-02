# estimate_TVPVAR.R
# Implements a Gibbs sampler for the TVP-VAR with stochastic
# volatility of Primiceri (2005); see Section 13.2 of the textbook.
#
# Model
#   y_t = X_t * beta_t + eps_t,   eps_t ~ N(0, Sigma_t),
#   Sigma_t^{-1} = L_t' * D_t^{-1} * L_t,
#   D_t = diag(exp(h_{1t}), ..., exp(h_{nt})),
# where L_t is unit-lower-triangular and l_t collects its m=n(n-1)/2
# free elements stacked equation by equation
#     l_t = (l_{2,1,t}, l_{3,1,t}, l_{3,2,t}, ..., l_{n,n-1,t})'.
# State equations
#   beta_t = beta_{t-1} + u_t^beta,    u_t^beta ~ N(0, Q),
#   l_t    = l_{t-1}    + u_t^l,       u_t^l    ~ N(0, S),
#   h_{it} = h_{i,t-1}  + u_{it}^h,    u_{it}^h ~ N(0, sigh2_i),
# with full Q, block-diagonal S = diag(S_1,...,S_{n-1}), and
# Gaussian priors on (beta_0, l_0, h_0).
#
# Requires: SURform.R, SVRW.R
#
# Inputs
#   Y      : (T_all) x n data, with the FIRST p ROWS used only as
#            initial lags. Effective sample length is T = T_all - p.
#   p      : number of VAR lags.
#   prior  : list with components
#       beta0_mean (nk), beta0_var  (nk x nk SPD)
#       l0_mean    (m),  l0_var     (m  x m  SPD)
#       h0_mean    (n),  h0_var     (n  x n  SPD)
#       Q_nu (scalar), Q_S0 (nk x nk SPD; full IW prior scale)
#       S_nu (length n-1 vector), S_S0 (length n-1 list, S_S0[[j]] is j x j SPD)
#       sigh2_nu (n), sigh2_S0 (n)
#   nsim, burnin : MCMC settings.
#
# Outputs (post-burn-in, returned in a list)
#   store_beta  : nsim x (T*nk)
#   store_l     : nsim x (T*m)
#   store_h     : nsim x T x n
#   store_Sig_t : nsim x (n*n) x T   (NULL if too large)
#   store_Q     : nsim x (nk*nk)
#   store_S     : nsim x (m*m)
#   store_sigh2 : nsim x n
#   store_beta0 : nsim x nk
#   store_l0    : nsim x m
#   store_h0    : nsim x n

suppressMessages(library(Matrix))
source("SURform.R")
source("SVRW.R")
# MATLAB's iwishrnd (Statistics Toolbox) has no base-R counterpart: draw
# X ~ IW(S, df) from the Bartlett decomposition, so that E[X] = S/(df-d-1).
iwishrnd <- function(S, df) {
    d <- nrow(S)
    C <- chol(S)                    # upper: S = t(C) %*% C
    A <- matrix(0, d, d)            # lower-triangular Bartlett factor
    diag(A) <- sqrt(rchisq(d, df - (1:d) + 1))
    if (d > 1)
        A[lower.tri(A)] <- rnorm(d*(d-1)/2)
    crossprod(forwardsolve(A, C))   # = (A^{-1} C)' (A^{-1} C)
}

estimate_TVPVAR <- function(Y, p, prior, nsim, burnin) {

Tall <- nrow(Y)
n    <- ncol(Y)
k  <- 1 + n*p
nk <- n*k
m  <- n*(n-1)/2
T  <- Tall - p

# construct X and y
Yeff <- Y[(p+1):Tall, , drop = FALSE]
X_tilde <- matrix(0, T, n*p)
for (i in 1:p) {
    X_tilde[, ((i-1)*n+1):(i*n)] <- Y[(p-i+1):(Tall-i), , drop = FALSE]
}
y <- as.vector(t(Yeff))
X <- SURform(cbind(rep(1, n*T), kronecker(X_tilde, rep(1, n))))

# T x T first-difference operator H and its Gram matrix HtH = H'*H
# The random-walk precision on a stacked (T*r)-vector of states with inner
# covariance A is kron(HtH, inv(A)).
H   <- Diagonal(T) - sparseMatrix(i = 2:T, j = 1:(T-1), x = rep(1, T-1),
                                  dims = c(T, T))
HtH <- crossprod(H)

# prior
beta0_mean <- as.numeric(prior$beta0_mean)
iVbeta0    <- solve(prior$beta0_var)
l0_mean    <- as.numeric(prior$l0_mean)
iVl0       <- solve(prior$l0_var)
h0_mean    <- as.numeric(prior$h0_mean)
iVh0       <- solve(prior$h0_var)
Q_nu       <- prior$Q_nu
Q_S0       <- prior$Q_S0
S_nu       <- prior$S_nu
S_S0       <- prior$S_S0
sigh2_nu   <- as.numeric(prior$sigh2_nu)
sigh2_S0   <- as.numeric(prior$sigh2_S0)

# linear indices of the free entries below the diagonal of L, eq-by-eq order
L_id <- matrix(1:(n^2), n, n)
L_id[!lower.tri(L_id)] <- 0
L_id <- t(L_id)
L_id <- L_id[L_id != 0]

# initialize chain at zero states (h = 0 implies sigma = 1) with tiny
# innovation covariances so the first beta draw is heavily smoothed.
l <- numeric(T*m)
h <- matrix(0, T, n)
Q <- 1e-4 * diag(nk)
sigh2 <- 1e-4 * rep(1, n)
S <- vector("list", n-1)
for (ii in 1:(n-1)) {
    S[[ii]] <- 1e-4 * diag(ii)
}

# initial conditions
beta0 <- beta0_mean
l0    <- l0_mean
h0    <- h0_mean

# storage
store_beta  <- matrix(0, nsim, T*nk)
store_l     <- matrix(0, nsim, T*m)
store_h     <- array(0, dim = c(nsim, T, n))
populate_Sig <- (n*n*T*nsim < 5e7)
if (populate_Sig) {
    store_Sig_t <- array(0, dim = c(nsim, n*n, T))
    # (row, column) positions of the n*n entries of each diagonal block of
    # blkdiag(Sigma_1, ..., Sigma_T), listed in column-major order within
    # blocks: this reproduces MATLAB's nonzeros(Sig_big) but is immune to a
    # structurally zero entry inside a block.
    blk_r <- rep(1:n, times = n)
    blk_c <- rep(1:n, each = n)
    Sig_I <- rep((0:(T-1))*n, each = n*n) + rep(blk_r, T)
    Sig_J <- rep((0:(T-1))*n, each = n*n) + rep(blk_c, T)
    Sig_ij <- cbind(Sig_I, Sig_J)
} else {
    store_Sig_t <- NULL
}
store_Q     <- matrix(0, nsim, nk*nk)
store_S     <- matrix(0, nsim, m*m)
store_sigh2 <- matrix(0, nsim, n)
store_beta0 <- matrix(0, nsim, nk)
store_l0    <- matrix(0, nsim, m)
store_h0    <- matrix(0, nsim, n)

# MCMC starts here
for (isim in 1:(nsim + burnin)) {
    # sample beta
    iSig_big <- build_iSig(l, h, n, T, L_id)
    iQ <- solve(Q)
    HiQH <- kronecker(HtH, iQ)
    rhs_b <- numeric(T*nk)
    rhs_b[1:nk] <- as.numeric(iQ %*% beta0)
    XiSig <- crossprod(X, iSig_big)
    # Kbeta is banded with bandwidth 2*nk-1; keep it sparse and never densify
    Kbeta <- forceSymmetric(HiQH + XiSig %*% X)
    Cb <- chol(Kbeta)   # upper factor: Kbeta = t(Cb) %*% Cb
    beta_hat <- solve(Cb, solve(t(Cb), rhs_b + as.numeric(XiSig %*% y)))
    beta <- as.numeric(beta_hat + solve(Cb, rnorm(T*nk)))

    # sample l
    eps_full <- as.numeric(y - X %*% beta)
    U <- t(matrix(eps_full, n, T))
    E_big <- build_E(U, n, T, L_id)
    iD <- Diagonal(x = exp(-as.vector(t(h))))
    # Block-diagonal S^{-1} (m x m), with blocks S_i^{-1}
    iSmat <- blockdiag_inv(S, m)
    HliSHl <- kronecker(HtH, iSmat)
    rhs_l <- numeric(T*m)
    rhs_l[1:m] <- as.numeric(iSmat %*% l0)
    EiD <- crossprod(E_big, iD)
    Kl <- forceSymmetric(HliSHl + EiD %*% E_big)
    Cl <- chol(Kl)
    l_hat <- solve(Cl, solve(t(Cl), rhs_l + as.numeric(EiD %*% eps_full)))
    l <- as.numeric(l_hat + solve(Cl, rnorm(T*m)))

    # sample h equation by equation (eps_full unchanged since beta fixed)
    Eorth <- t(matrix(as.numeric(build_L(l, n, T, L_id) %*% eps_full), n, T))
    ystar <- log(Eorth^2 + 1e-4)
    for (ii in 1:n) {
        h[, ii] <- SVRW(ystar[, ii], h[, ii], h0[ii], sigh2[ii])
    }

    # sample beta0, l0, h0 (with diffuse priors); reuse iQ and iSmat
    Kb0 <- iVbeta0 + iQ
    Cb0 <- chol(Kb0)
    b0_hat <- backsolve(Cb0, backsolve(Cb0,
        as.numeric(iVbeta0 %*% beta0_mean + iQ %*% beta[1:nk]),
        transpose = TRUE))
    beta0 <- as.numeric(b0_hat + backsolve(Cb0, rnorm(nk)))

    Kl0 <- iVl0 + as.matrix(iSmat)
    Cl0 <- chol(Kl0)
    l0_hat <- backsolve(Cl0, backsolve(Cl0,
        as.numeric(iVl0 %*% l0_mean + iSmat %*% l[1:m]), transpose = TRUE))
    l0 <- as.numeric(l0_hat + backsolve(Cl0, rnorm(m)))

    isigh2 <- diag(1/sigh2, n, n)
    Kh0 <- iVh0 + isigh2
    Ch0 <- chol(Kh0)
    h0_hat <- backsolve(Ch0, backsolve(Ch0,
        as.numeric(iVh0 %*% h0_mean + isigh2 %*% h[1, ]), transpose = TRUE))
    h0 <- as.numeric(h0_hat + backsolve(Ch0, rnorm(n)))

    # sample Q, S, sigh2
    bmat <- matrix(beta, nk, T)
    db <- bmat - cbind(beta0, bmat[, 1:(T-1), drop = FALSE])
    Q <- iwishrnd(Q_S0 + tcrossprod(db), Q_nu + T)
    lmat <- matrix(l, m, T)
    dl <- lmat - cbind(l0, lmat[, 1:(T-1), drop = FALSE])
    off <- 0
    for (ii in 1:(n-1)) {
        di <- ii
        idx <- (off+1):(off+di)
        DLi <- dl[idx, , drop = FALSE]
        S[[ii]] <- iwishrnd(S_S0[[ii]] + tcrossprod(DLi), S_nu[ii] + T)
        off <- off + di
    }

    dh <- h - rbind(h0, h[1:(T-1), , drop = FALSE])
    # R's rgamma takes shape and scale
    sigh2 <- 1/rgamma(n, shape = sigh2_nu + T/2,
                      scale = 1/(sigh2_S0 + 0.5*colSums(dh^2)))

    if (isim > burnin) {
        isave <- isim - burnin
        store_beta[isave, ] <- beta
        store_l[isave, ]    <- l
        store_h[isave, , ]  <- h
        store_Q[isave, ]    <- as.vector(Q)
        Sfull <- matrix(0, m, m)
        off <- 0
        for (ii in 1:(n-1)) {
            di <- ii
            Sfull[(off+1):(off+di), (off+1):(off+di)] <- S[[ii]]
            off <- off + di
        }
        store_S[isave, ]     <- as.vector(Sfull)
        store_sigh2[isave, ] <- sigh2
        store_beta0[isave, ] <- beta0
        store_l0[isave, ]    <- l0
        store_h0[isave, ]    <- h0
        if (populate_Sig) {
            L_big <- build_L(l, n, T, L_id)
            D_sqrt <- Diagonal(x = exp(as.vector(t(h))/2))
            M_big <- solve(L_big, D_sqrt)     # M_t = L_t \ D_t^{1/2}
            Sig_big <- M_big %*% t(M_big)     # blkdiag(Sigma_1, ..., Sigma_T)
            store_Sig_t[isave, , ] <- matrix(Sig_big[Sig_ij], n*n, T)
        }
    }

    if (isim %% 1000 == 0) {
        cat(sprintf("estimate_TVPVAR: iteration %d / %d\n",
                    isim, nsim + burnin))
        flush.console()
    }
}

list(store_beta = store_beta, store_l = store_l, store_h = store_h,
     store_Sig_t = store_Sig_t, store_Q = store_Q, store_S = store_S,
     store_sigh2 = store_sigh2, store_beta0 = store_beta0,
     store_l0 = store_l0, store_h0 = store_h0)
}

#======================================================================
build_iSig <- function(l, h, n, T, L_id) {
    # Assemble sparse blkdiag(L_t' D_t^{-1} L_t, t=1..T), size (n*T) x (n*T).
    L_big <- build_L(l, n, T, L_id)
    iD <- Diagonal(x = exp(-as.vector(t(h))))
    crossprod(L_big, iD %*% L_big)
}

#======================================================================
build_L <- function(l, n, T, L_id) {
    # Assemble sparse L_big = blkdiag(L_1, ..., L_T), size (n*T) x (n*T),
    # where each L_t is unit lower-triangular with free entries given by
    # l_t (the t-th block of length m of the stacked vector l).
    m <- n*(n-1)/2

    # Position of each free l-entry within an n x n block
    r_pos <- (L_id - 1) %% n + 1          # row in {2, ..., n}
    c_pos <- (L_id - r_pos)/n + 1         # col in {1, ..., n-1}

    # Off-diagonal triplets of L_big = blkdiag(L_1, ..., L_T)
    t_off_n <- (0:(T-1))*n                # length T
    I <- outer(t_off_n, r_pos, "+")       # T x m
    J <- outer(t_off_n, c_pos, "+")       # T x m
    V <- t(matrix(l, m, T))               # T x m; row t is l_t'

    # Sparse L_big: unit diagonal in each block plus the off-diagonal entries
    diag_idx <- 1:(n*T)
    sparseMatrix(i = c(diag_idx, as.vector(I)),
                 j = c(diag_idx, as.vector(J)),
                 x = c(rep(1, n*T), as.vector(V)), dims = c(n*T, n*T))
}

#======================================================================
build_E <- function(U, n, T, L_id) {
    # Assemble sparse E (n*T x m*T) block-diagonal with the n x m block
    # E_t in slot t. E_t maps l_t to L_t * eps_t - eps_t, with row r_pos
    # and column pos carrying the entry -eps_{c_pos(pos), t}.
    m <- n*(n-1)/2

    # Position of each free l-entry within an n x n block
    r_pos <- (L_id - 1) %% n + 1          # row in {2, ..., n}
    c_pos <- (L_id - r_pos)/n + 1         # col in {1, ..., n-1}

    # Triplets of E_big = blkdiag(E_1, ..., E_T)
    t_off_n <- (0:(T-1))*n                # length T
    t_off_m <- (0:(T-1))*m                # length T
    I <- outer(t_off_n, r_pos, "+")       # T x m
    J <- outer(t_off_m, 1:m, "+")         # T x m
    V <- -U[, c_pos, drop = FALSE]        # T x m

    sparseMatrix(i = as.vector(I), j = as.vector(J), x = as.vector(V),
                 dims = c(n*T, m*T))
}

#======================================================================
blockdiag_inv <- function(S, m) {
    # Sparse block-diagonal inverse of the list S: S[[i]] is i x i SPD;
    # total size m = sum_{i=1}^{n-1} i.
    nblocks <- length(S)
    total_nnz <- sum((1:nblocks)^2)
    I <- numeric(total_nnz)
    J <- numeric(total_nnz)
    V <- numeric(total_nnz)
    ptr <- 0
    off <- 0
    for (i in 1:nblocks) {
        di <- i
        iSi <- solve(S[[i]])
        nz <- di*di
        I[(ptr+1):(ptr+nz)] <- off + rep(1:di, times = di)
        J[(ptr+1):(ptr+nz)] <- off + rep(1:di, each = di)
        V[(ptr+1):(ptr+nz)] <- as.vector(iSi)
        ptr <- ptr + nz
        off <- off + di
    }
    sparseMatrix(i = I, j = J, x = V, dims = c(m, m))
}
