# pred_VAR_OISV.R
# Computes the one-step-ahead posterior predictive mean
# and log predictive likelihood of y_{t, var_idx} from the VAR with
# order-invariant stochastic volatility,
#     y_t = x_t' A  + eps_t,    eps_t ~ N(0, Sig_t),
#     Sig_t^{-1} = B_0' D_t^{-1} B_0,  D_t = diag(exp(h_{1t}),...,exp(h_{nt})),
#     h_{i,t} = phi_i*h_{i,t-1} + u_{i,t}^h,    u_{i,t}^h ~ N(0, sigh2_i),
#     h_{i,1} ~ N(0, sigh2_i/(1 - phi_i^2)),
# with the independent Minnesota prior beta ~ N(beta0, diag(V_Minn)),
# N(b_{0,i}, V_b) prior on each row b_i of B_0,
# TN_{(-1,1)}(phi_0, V_phi) prior on phi_i, and
# IG(nu_h, S_h) prior on each sigh2_i. Five-block Gibbs sampler
# following Section 13.1.3:
#   1) B_0 row by row via the Waggoner-Zha-Villani approach
#   2) beta given B_0 and h (Gaussian regression)
#   3) h_{i,1:T} via SVAR1 (stationary AR(1) auxiliary mixture sampler)
#   4) phi_i via independence-chain Metropolis-Hastings
#   5) sigh2_i ~ IG
#
# Requires: SURform2.R, sample_B0.R, SVAR1.R, sample_SVAR1para.R
#
# Inputs:
# Yt:       T x n matrix of observations up to time t-1
# Z:        T x k matrix of regressors (intercept + p lags), k = 1+n*p
# xt:       length-k regressor row for forecasting y_t
# beta0:    (n*k)-vector Minnesota prior mean of beta
# V_Minn:   (n*k)-vector of diagonal entries of the Minnesota prior covariance
# nsim:     scalar; number of post-burnin MCMC draws
# burnin:   scalar; number of burnin draws
# var_idx:  scalar; index of the variable to forecast
# yreal:    scalar; realized value of y_{t, var_idx} (for LPL)
#
# Outputs (returned in a list):
# pf:   posterior predictive mean of y_{t, var_idx}
# lpl:  log of the predictive density at yreal, computed as the
#       log-mean-exp over MCMC draws of mixture-of-Gaussians weights

suppressMessages(library(Matrix))
source("SURform2.R")
source("sample_B0.R")
source("SVAR1.R")
source("sample_SVAR1para.R")

pred_VAR_OISV <- function(Yt, Z, xt, beta0, V_Minn, nsim, burnin,
                          var_idx, yreal) {

    T <- nrow(Yt)
    n <- ncol(Yt)
    k <- ncol(Z)
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
        }
    }

    # aggregate across draws
    pf <- mean(store_mu)
    log_w <- -0.5*log(2*pi*store_var) - 0.5*(yreal - store_mu)^2/store_var
    lpl <- log(mean(exp(log_w - max(log_w)))) + max(log_w)

    list(pf = pf, lpl = lpl)
}
