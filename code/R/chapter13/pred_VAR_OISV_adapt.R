# pred_VAR_OISV_adapt.R
# Adaptive-Minnesota version of pred_VAR_OISV.R (order-invariant
# stochastic volatility). Identical five-block Gibbs sampler, plus an
# extra Gibbs step that estimates the Minnesota shrinkage
# hyperparameters kappa2 (own-lag) and kappa3 (cross-lag) from their
# conjugate IG full conditionals; the intercept prior variance (kappa1)
# is held fixed.
#
# Requires: SURform2.R, sample_B0.R, SVAR1.R, sample_SVAR1para.R
#
# Extra input:
# s2_hat:   n-vector of univariate AR(p) residual variances (from Minn_indep)
# Extra outputs:
# k2m, k3m: posterior means of kappa2, kappa3

suppressMessages(library(Matrix))
source("SURform2.R")
source("sample_B0.R")
source("SVAR1.R")
source("sample_SVAR1para.R")

pred_VAR_OISV_adapt <- function(Yt, Z, xt, beta0, V_Minn, s2_hat,
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

    # OI-SV-specific priors
    b0_B0 <- diag(n)   # prior mean: row i is e_i
    iV_B0 <- diag(n)   # prior precision of each row (V_b = I)
    phi_0 <- 0.9; V_phi <- 0.2^2
    nu_h  <- 3*rep(1, n); S_h <- 0.1*rep(1, n)

    # SUR-form regressors and time-stacked observations
    X <- SURform2(Z, n)
    y <- as.vector(t(Yt))

    # initialize the Gibbs sampler at OLS
    A <- qr.solve(Z, Yt)
    beta <- as.vector(A)
    E <- Yt - Z %*% A
    h0 <- log(diag(crossprod(E)/T))
    h  <- matrix(h0, T, n, byrow = TRUE)
    sigh2 <- 0.1*rep(1, n)
    phi   <- 0.9*rep(1, n)
    B0    <- diag(n)

    store_mu  <- numeric(nsim)
    store_var <- numeric(nsim)
    store_k2  <- numeric(nsim)
    store_k3  <- numeric(nsim)

    for (isim in 1:(nsim + burnin)) {
        # sample B_0 row by row via Waggoner-Zha-Villani
        A <- matrix(beta, k, n)
        E <- Yt - Z %*% A
        B0 <- sample_B0(B0, E, h, b0_B0, iV_B0)

        # sample beta
        bigB0 <- kronecker(Diagonal(T), Matrix(B0, sparse = TRUE))
        iD <- Diagonal(x = as.vector(t(1/exp(h))))
        iSig <- crossprod(bigB0, iD %*% bigB0)
        XiSig <- crossprod(X, iSig)
        Kbeta <- as.matrix(iVbeta + XiSig %*% X)
        Cbeta <- chol(Kbeta)   # upper factor: Kbeta = t(Cbeta) %*% Cbeta
        rhs <- as.numeric(iVbeta %*% beta0 + XiSig %*% y)
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

        # sample h equation by equation
        A <- matrix(beta, k, n)
        E <- Yt - Z %*% A
        Eorth <- E %*% t(B0)
        ystar <- log(Eorth^2 + 1e-4)
        for (i in 1:n) {
            h[, i] <- SVAR1(ystar[, i], h[, i], 0, phi[i], sigh2[i])
        }

        # sample phi and sigh2
        para <- sample_SVAR1para(h, phi, sigh2, phi_0, V_phi, nu_h, S_h)
        phi   <- para$phi
        sigh2 <- para$sigh2

        if (isim > burnin) {
            isave <- isim - burnin
            mu_full <- as.numeric(xt %*% A)
            # forecast h_{T+1} via stationary AR(1)
            h_tp1 <- phi*h[T, ] + sqrt(sigh2)*rnorm(n)
            iB0 <- solve(B0)
            Sig_tp1 <- iB0 %*% diag(exp(h_tp1), n, n) %*% t(iB0)
            store_mu[isave]  <- mu_full[var_idx]
            store_var[isave] <- Sig_tp1[var_idx, var_idx]
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
