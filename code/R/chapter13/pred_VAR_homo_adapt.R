# pred_VAR_homo_adapt.R
# Adaptive-Minnesota version of pred_VAR_homo.R. Identical two-block
# Gibbs sampler for the homoskedastic VAR, except that the Minnesota
# shrinkage hyperparameters kappa2 (own-lag) and kappa3 (cross-lag) are
# ESTIMATED from the data via an additional Gibbs step. Given the base
# divisors d (d_own = 1/l^2, d_cross = s_i^2/(l^2 s_j^2)) so that
#   Var(own a_{l,ii})    = kappa2 * d_own,
#   Var(cross a_{l,ij})  = kappa3 * d_cross,
# and independent IG(nu_k, S_k) priors on kappa2, kappa3 with zero prior
# mean on the coefficients, the full conditionals are conjugate:
#   kappa2 | beta ~ IG(nu_k2 + n_own/2,   S_k2 + 0.5*sum_own  b_j^2/d_j),
#   kappa3 | beta ~ IG(nu_k3 + n_cross/2, S_k3 + 0.5*sum_cross b_j^2/d_j).
# The intercept prior variance (kappa1) is held fixed.
#
# Extra input:
# s2_hat:   n-vector of univariate AR(p) residual variances (from Minn_indep)
# Extra outputs:
# k2m, k3m: posterior means of kappa2, kappa3

suppressMessages(library(Matrix))
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

pred_VAR_homo_adapt <- function(Yt, Z, xt, beta0, V_Minn, s2_hat,
                                nsim, burnin, var_idx, yreal) {

    T <- nrow(Yt)
    n <- ncol(Yt)
    k <- ncol(Z)
    p <- (k - 1)/n

    # adaptive-Minnesota IG priors on kappa2, kappa3 (prior means 0.2^2,
    # 0.2^2/4)
    nu_k2 <- 3; S_k2 <- 2*0.2^2
    nu_k3 <- 3; S_k3 <- 2*0.2^2/4

    # own- vs cross-lag positions and base divisors (mirror Minn_indep loop)
    n_own   <- n*p
    n_cross <- n*(n-1)*p
    own_idx   <- numeric(n_own);   d_own   <- numeric(n_own)
    cross_idx <- numeric(n_cross); d_cross <- numeric(n_cross)
    co <- 0; cc <- 0; count <- 1
    for (i in 1:n) {
        count <- count + 1               # intercept position (kappa1, fixed)
        for (l in 1:p) {
            for (j in 1:n) {
                if (i == j) {
                    co <- co + 1; own_idx[co] <- count; d_own[co] <- 1/l^2
                } else {
                    cc <- cc + 1; cross_idx[cc] <- count
                    d_cross[cc] <- s2_hat[i]/(l^2*s2_hat[j])
                }
                count <- count + 1
            }
        }
    }

    iVbeta <- Diagonal(x = 1/V_Minn)

    # inverse-Wishart prior on Sig
    nu0 <- n + 2
    S0  <- diag(n)

    # precompute
    ZZ <- crossprod(Z)
    ZY <- crossprod(Z, Yt)

    # initialize the Gibbs sampler at OLS
    A <- qr.solve(Z, Yt)
    beta <- as.vector(A)
    E <- Yt - Z %*% A
    Sig <- crossprod(E)/T
    iSig <- solve(Sig)

    store_mu  <- numeric(nsim)
    store_var <- numeric(nsim)
    store_k2  <- numeric(nsim)
    store_k3  <- numeric(nsim)

    for (isim in 1:(nsim + burnin)) {
        # sample beta
        Kbeta <- as.matrix(iVbeta + kronecker(iSig, ZZ))
        Cbeta <- chol(Kbeta)   # upper factor: Kbeta = t(Cbeta) %*% Cbeta
        rhs <- as.numeric(iVbeta %*% beta0) + as.vector(ZY %*% iSig)
        beta_hat <- backsolve(Cbeta, backsolve(Cbeta, rhs, transpose = TRUE))
        beta <- as.numeric(beta_hat + backsolve(Cbeta, rnorm(n*k)))

        # sample kappa2, kappa3 and rebuild V_Minn / iVbeta (intercepts fixed)
        b_own   <- beta[own_idx]   - beta0[own_idx]
        b_cross <- beta[cross_idx] - beta0[cross_idx]
        # R's rgamma takes shape and scale
        kappa2 <- 1/rgamma(1, shape = nu_k2 + n_own/2,
                           scale = 1/(S_k2 + 0.5*sum(b_own^2/d_own)))
        kappa3 <- 1/rgamma(1, shape = nu_k3 + n_cross/2,
                           scale = 1/(S_k3 + 0.5*sum(b_cross^2/d_cross)))
        V_Minn[own_idx]   <- kappa2 * d_own
        V_Minn[cross_idx] <- kappa3 * d_cross
        iVbeta <- Diagonal(x = 1/V_Minn)

        # sample Sig
        A <- matrix(beta, k, n)
        E <- Yt - Z %*% A
        Sig <- iwishrnd(S0 + crossprod(E), nu0 + T)
        iSig <- solve(Sig)

        if (isim > burnin) {
            isave <- isim - burnin
            mu_full <- as.numeric(xt %*% A)
            store_mu[isave]  <- mu_full[var_idx]
            store_var[isave] <- Sig[var_idx, var_idx]
            store_k2[isave]  <- kappa2
            store_k3[isave]  <- kappa3
        }
    }

    # aggregate across draws
    pf <- mean(store_mu)
    log_w <- -0.5*log(2*pi*store_var) - 0.5*(yreal - store_mu)^2/store_var
    lpl <- log(mean(exp(log_w - max(log_w)))) + max(log_w)
    k2m <- mean(store_k2)
    k3m <- mean(store_k3)

    list(pf = pf, lpl = lpl, k2m = k2m, k3m = k3m)
}
