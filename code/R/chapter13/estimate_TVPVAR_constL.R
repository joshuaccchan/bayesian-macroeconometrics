# estimate_TVPVAR_constL.R
# Gibbs sampler for the TVP-VAR with stochastic volatility in which the
# lower-triangular factor L in Sigma_t^{-1} = L' * D_t^{-1} * L is held
# CONSTANT over time (L_t = L for all t). This is the Cogley-Sargent (2005)
# specification: only the VAR coefficients beta_t and the log-volatilities
# h_t are time-varying, while the contemporaneous relations L are constant.
#
# Model
#   y_t = X_t * beta_t + eps_t,   eps_t ~ N(0, Sigma_t),
#   Sigma_t^{-1} = L' * D_t^{-1} * L,     (L constant, unit lower triangular)
#   D_t = diag(exp(h_{1t}), ..., exp(h_{nt})),
# where l collects the m = n(n-1)/2 free elements of L stacked
# equation by equation  l = (l_{2,1}, l_{3,1}, l_{3,2}, ..., l_{n,n-1})'.
# State equations
#   beta_t = beta_{t-1} + u_t^beta,   u_t^beta ~ N(0, Q),
#   h_{it} = h_{i,t-1}  + u_{it}^h,   u_{it}^h ~ N(0, sigh2_i),
# with full Q and Gaussian priors on (beta_0, h_0). The constant l has a
# fixed Gaussian prior l ~ N(m_l, V_l).
#
# Relative to estimate_TVPVAR.R, Step 2 (time-varying l path with random-walk
# innovation covariance S) is replaced by a single constant-l Gaussian draw,
# and the S block / l_0 initial condition are removed. Blocks 1 (beta RW),
# 3 (h RW SV), 4 (beta_0, h_0), and the Q, sigh2 hyperparameter draws are
# unchanged from estimate_TVPVAR.R.
#
# Requires: SURform.R, SVRW.R
# The helpers build_iSig and build_L below are byte-identical to the ones in
# estimate_TVPVAR.R (as in the MATLAB originals, where they are local
# functions of each file), so sourcing both files is harmless.
#
# Inputs
#   Y      : (T_all) x n data, with the FIRST p ROWS used only as
#            initial lags. Effective sample length is T = T_all - p.
#   p      : number of VAR lags.
#   prior  : list with components
#       beta0_mean (nk), beta0_var  (nk x nk SPD)
#       l0_mean    (m),  l0_var     (m  x m  SPD)   # used as (m_l, V_l)
#       h0_mean    (n),  h0_var     (n  x n  SPD)
#       Q_nu (scalar), Q_S0 (nk x nk SPD; full IW prior scale)
#       sigh2_nu (n), sigh2_S0 (n)
#     (Components S_nu, S_S0 are ignored if present.)
#   nsim, burnin : MCMC settings.
#
# Outputs (post-burn-in, returned in a list)
#   store_beta  : nsim x (T*nk)
#   store_l     : nsim x m           (constant l for each draw)
#   store_h     : nsim x T x n
#   store_Sig_t : NULL  (residual std devs recomputed from store_l, store_h)
#   store_Q     : nsim x (nk*nk)
#   store_sigh2 : nsim x n
#   store_beta0 : nsim x nk
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

estimate_TVPVAR_constL <- function(Y, p, prior, nsim, burnin) {

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
H   <- Diagonal(T) - sparseMatrix(i = 2:T, j = 1:(T-1), x = rep(1, T-1),
                                  dims = c(T, T))
HtH <- crossprod(H)

# prior
beta0_mean <- as.numeric(prior$beta0_mean)
iVbeta0    <- solve(prior$beta0_var)
l_mean     <- as.numeric(prior$l0_mean)      # prior mean m_l of constant l
iVl        <- solve(prior$l0_var)            # prior precision V_l^{-1}
h0_mean    <- as.numeric(prior$h0_mean)
iVh0       <- solve(prior$h0_var)
Q_nu       <- prior$Q_nu
Q_S0       <- prior$Q_S0
sigh2_nu   <- as.numeric(prior$sigh2_nu)
sigh2_S0   <- as.numeric(prior$sigh2_S0)

# linear indices of the free entries below the diagonal of L, eq-by-eq order
L_id <- matrix(1:(n^2), n, n)
L_id[!lower.tri(L_id)] <- 0
L_id <- t(L_id)
L_id <- L_id[L_id != 0]

# initialize chain at zero states (h = 0 implies sigma = 1) with tiny
# innovation covariances so the first beta draw is heavily smoothed.
l <- l_mean                 # constant l initialized at prior mean
h <- matrix(0, T, n)
Q <- 1e-4 * diag(nk)
sigh2 <- 1e-4 * rep(1, n)

# initial conditions
beta0 <- beta0_mean
h0    <- h0_mean

# storage
store_beta  <- matrix(0, nsim, T*nk)
store_l     <- matrix(0, nsim, m)
store_h     <- array(0, dim = c(nsim, T, n))
store_Sig_t <- NULL
store_Q     <- matrix(0, nsim, nk*nk)
store_sigh2 <- matrix(0, nsim, n)
store_beta0 <- matrix(0, nsim, nk)
store_h0    <- matrix(0, nsim, n)

# MCMC starts here
for (isim in 1:(nsim + burnin)) {
    # replicate the constant l across all T blocks for the sparse builders
    l_rep <- rep(l, T)

    # ---- Block 1: sample beta (unchanged) ----
    iSig_big <- build_iSig(l_rep, h, n, T, L_id)
    iQ <- solve(Q)
    HiQH <- kronecker(HtH, iQ)
    rhs_b <- numeric(T*nk)
    rhs_b[1:nk] <- as.numeric(iQ %*% beta0)
    XiSig <- crossprod(X, iSig_big)
    Kbeta <- forceSymmetric(HiQH + XiSig %*% X)
    Cb <- chol(Kbeta)   # upper factor: Kbeta = t(Cb) %*% Cb
    beta_hat <- solve(Cb, solve(t(Cb), rhs_b + as.numeric(XiSig %*% y)))
    beta <- as.numeric(beta_hat + solve(Cb, rnorm(T*nk)))

    # ---- Block 2: sample CONSTANT l ----
    # eps_t = E_t l + u_t, u_t ~ N(0, D_t), stacked over all t:
    #   (l | .) ~ N(lhat, Kl^{-1}),  Kl = V_l^{-1} + E' D^{-1} E,
    #   lhat = Kl^{-1} (V_l^{-1} m_l + E' D^{-1} eps).
    eps_full <- as.numeric(y - X %*% beta)
    U <- t(matrix(eps_full, n, T))
    E_big <- build_E_const(U, n, T, L_id)
    iD <- Diagonal(x = exp(-as.vector(t(h))))
    EiD <- crossprod(E_big, iD)
    Kl  <- as.matrix(iVl + EiD %*% E_big)
    Cl  <- chol(Kl)
    l_hat <- backsolve(Cl, backsolve(Cl,
        as.numeric(iVl %*% l_mean + EiD %*% eps_full), transpose = TRUE))
    l <- as.numeric(l_hat + backsolve(Cl, rnorm(m)))

    # ---- Block 3: sample h equation by equation (unchanged) ----
    l_rep <- rep(l, T)
    Eorth <- t(matrix(as.numeric(build_L(l_rep, n, T, L_id) %*% eps_full),
                      n, T))
    ystar <- log(Eorth^2 + 1e-4)
    for (ii in 1:n) {
        h[, ii] <- SVRW(ystar[, ii], h[, ii], h0[ii], sigh2[ii])
    }

    # ---- Block 4: sample beta0, h0 (l0 removed) ----
    Kb0 <- iVbeta0 + iQ
    Cb0 <- chol(Kb0)
    b0_hat <- backsolve(Cb0, backsolve(Cb0,
        as.numeric(iVbeta0 %*% beta0_mean + iQ %*% beta[1:nk]),
        transpose = TRUE))
    beta0 <- as.numeric(b0_hat + backsolve(Cb0, rnorm(nk)))

    isigh2 <- diag(1/sigh2, n, n)
    Kh0 <- iVh0 + isigh2
    Ch0 <- chol(Kh0)
    h0_hat <- backsolve(Ch0, backsolve(Ch0,
        as.numeric(iVh0 %*% h0_mean + isigh2 %*% h[1, ]), transpose = TRUE))
    h0 <- as.numeric(h0_hat + backsolve(Ch0, rnorm(n)))

    # ---- Block 5: sample Q, sigh2 (S removed) ----
    bmat <- matrix(beta, nk, T)
    db <- bmat - cbind(beta0, bmat[, 1:(T-1), drop = FALSE])
    Q <- iwishrnd(Q_S0 + tcrossprod(db), Q_nu + T)

    dh <- h - rbind(h0, h[1:(T-1), , drop = FALSE])
    # R's rgamma takes shape and scale
    sigh2 <- 1/rgamma(n, shape = sigh2_nu + T/2,
                      scale = 1/(sigh2_S0 + 0.5*colSums(dh^2)))

    if (isim > burnin) {
        isave <- isim - burnin
        store_beta[isave, ]  <- beta
        store_l[isave, ]     <- l
        store_h[isave, , ]   <- h
        store_Q[isave, ]     <- as.vector(Q)
        store_sigh2[isave, ] <- sigh2
        store_beta0[isave, ] <- beta0
        store_h0[isave, ]    <- h0
    }

    if (isim %% 1000 == 0) {
        cat(sprintf("estimate_TVPVAR_constL: iteration %d / %d\n",
                    isim, nsim + burnin))
        flush.console()
    }
}

list(store_beta = store_beta, store_l = store_l, store_h = store_h,
     store_Sig_t = store_Sig_t, store_Q = store_Q,
     store_sigh2 = store_sigh2, store_beta0 = store_beta0,
     store_h0 = store_h0)
}

#======================================================================
build_iSig <- function(l, h, n, T, L_id) {
    # Assemble sparse blkdiag(L' D_t^{-1} L, t=1..T), size (n*T) x (n*T).
    L_big <- build_L(l, n, T, L_id)
    iD <- Diagonal(x = exp(-as.vector(t(h))))
    crossprod(L_big, iD %*% L_big)
}

#======================================================================
build_L <- function(l, n, T, L_id) {
    # Assemble sparse L_big = blkdiag(L_1, ..., L_T), size (n*T) x (n*T),
    # where each L_t is unit lower-triangular with free entries given by
    # the t-th block of length m of the stacked vector l.
    m <- n*(n-1)/2

    r_pos <- (L_id - 1) %% n + 1          # row in {2, ..., n}
    c_pos <- (L_id - r_pos)/n + 1         # col in {1, ..., n-1}

    t_off_n <- (0:(T-1))*n                # length T
    I <- outer(t_off_n, r_pos, "+")       # T x m
    J <- outer(t_off_n, c_pos, "+")       # T x m
    V <- t(matrix(l, m, T))               # T x m; row t is l_t'

    diag_idx <- 1:(n*T)
    sparseMatrix(i = c(diag_idx, as.vector(I)),
                 j = c(diag_idx, as.vector(J)),
                 x = c(rep(1, n*T), as.vector(V)), dims = c(n*T, n*T))
}

#======================================================================
build_E_const <- function(U, n, T, L_id) {
    # Assemble sparse E (n*T x m) mapping the CONSTANT l to the stacked
    # n*T vector with block E_t in rows (t-1)*n+1 : t*n. All blocks share
    # the SAME m columns (unlike the time-varying case), so E_t l = L*eps_t
    # - eps_t. Row r_pos, column pos carries -eps_{c_pos(pos), t}.
    m <- n*(n-1)/2

    r_pos <- (L_id - 1) %% n + 1          # row in {2, ..., n}
    c_pos <- (L_id - r_pos)/n + 1         # col in {1, ..., n-1}

    t_off_n <- (0:(T-1))*n                # length T
    I <- outer(t_off_n, r_pos, "+")       # T x m
    J <- matrix(1:m, T, m, byrow = TRUE)  # T x m; columns fixed (no t offset)
    V <- -U[, c_pos, drop = FALSE]        # T x m

    sparseMatrix(i = as.vector(I), j = as.vector(J), x = as.vector(V),
                 dims = c(n*T, m))
}
