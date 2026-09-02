# pred_VAR_SV_adapt.R
# Adaptive-Minnesota version of pred_VAR_SV.R (Cholesky stochastic
# volatility). Identical five-block Gibbs sampler, plus an extra Gibbs
# step that estimates the Minnesota shrinkage hyperparameters kappa2
# (own-lag) and kappa3 (cross-lag) from their conjugate IG full
# conditionals; the intercept prior variance (kappa1) is held fixed.
#
# Requires: SURform2.R, SVRW.R
#
# Extra input:
# s2_hat:   n-vector of univariate AR(p) residual variances (from Minn_indep)
# Extra outputs:
# k2m, k3m: posterior means of kappa2, kappa3

suppressMessages(library(Matrix))
source("SURform2.R")
source("SVRW.R")

pred_VAR_SV_adapt <- function(Yt, Z, xt, beta0, V_Minn, s2_hat,
                              nsim, burnin, var_idx, yreal) {

    T <- nrow(Yt)
    n <- ncol(Yt)
    k <- ncol(Z)
    p <- (k - 1)/n
    m <- n*(n - 1)/2

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

    # SV-specific priors
    l0   <- numeric(m);          iVl  <- diag(m)
    m_h0 <- numeric(n);          iVh0 <- diag(n)/10
    nu_h <- 3*rep(1, n);         S_h  <- 0.1*rep(1, n)

    # indices of the strictly lower-triangular elements of L in row-major order
    L_id <- numeric(m)
    ii <- 0
    for (i in 2:n) {
        for (j in 1:(i-1)) {
            ii <- ii + 1
            L_id[ii] <- i + (j-1)*n
        }
    }

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
    l_vec <- numeric(m)
    L <- diag(n)

    store_mu  <- numeric(nsim)
    store_var <- numeric(nsim)
    store_k2  <- numeric(nsim)
    store_k3  <- numeric(nsim)

    for (isim in 1:(nsim + burnin)) {
        # sample beta
        L[L_id] <- l_vec
        bigL <- kronecker(Diagonal(T), Matrix(L, sparse = TRUE))
        iD <- Diagonal(x = as.vector(t(1/exp(h))))
        iSig <- crossprod(bigL, iD %*% bigL)
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
        Eorth <- E %*% t(L)
        ystar <- log(Eorth^2 + 1e-4)
        for (i in 1:n) {
            h[, i] <- SVRW(ystar[, i], h[, i], h0[i], sigh2[i])
        }

        # sample l_vec from the regression eps_t = E_t * l + eta_t
        Em <- matrix(0, T*n, m)
        cE <- 0
        for (ii in 1:(n-1)) {
            Em[seq(ii+1, T*n, by = n), (cE+1):(cE+ii)] <- -E[, 1:ii, drop = FALSE]
            cE <- cE + ii
        }
        iD <- Diagonal(x = as.vector(t(1/exp(h))))
        Kl <- as.matrix(iVl + crossprod(Em, as.matrix(iD %*% Em)))
        Cl <- chol(Kl)
        rhs_l <- as.numeric(iVl %*% l0) +
            as.numeric(crossprod(Em, as.numeric(iD %*% as.vector(t(E)))))
        l_hat <- backsolve(Cl, backsolve(Cl, rhs_l, transpose = TRUE))
        l_vec <- as.numeric(l_hat + backsolve(Cl, rnorm(m)))

        # sample sigh2
        e2 <- (h - rbind(h0, h[1:(T-1), , drop = FALSE]))^2
        sigh2 <- 1/rgamma(n, shape = nu_h + T/2,
                          scale = 1/(S_h + colSums(e2)/2))

        # sample h0
        Kh0 <- iVh0 + diag(1/sigh2, n, n)
        Ch0 <- chol(Kh0)
        rhs_h0 <- as.numeric(iVh0 %*% m_h0) + h[1, ]/sigh2
        h0_hat <- backsolve(Ch0, backsolve(Ch0, rhs_h0, transpose = TRUE))
        h0 <- as.numeric(h0_hat + backsolve(Ch0, rnorm(n)))

        if (isim > burnin) {
            isave <- isim - burnin
            mu_full <- as.numeric(xt %*% A)
            # forecast h_{T+1} via random walk and the implied Sig_{T+1}
            h_tp1 <- h[T, ] + sqrt(sigh2)*rnorm(n)
            L[L_id] <- l_vec
            invL <- solve(L)
            Sig_tp1 <- invL %*% diag(exp(h_tp1), n, n) %*% t(invL)
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
