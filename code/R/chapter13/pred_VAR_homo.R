# pred_VAR_homo.R
# Computes the one-step-ahead posterior predictive mean
# and log predictive likelihood of y_{t, var_idx} from the
# homoskedastic VAR
#     y_t = A' x_t + eps_t,   eps_t ~ N(0, Sig),
# with the independent Minnesota prior beta ~ N(beta0, diag(V_Minn))
# and the inverse-Wishart prior Sig ~ IW(nu0, S0). Two-block Gibbs
# sampler. The point forecast and LPL are aggregated within the
# function as the posterior predictive mean and the log-mean-exp of
# the per-draw Gaussian densities.
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

pred_VAR_homo <- function(Yt, Z, xt, beta0, V_Minn, nsim, burnin,
                          var_idx, yreal) {

    T <- nrow(Yt)
    n <- ncol(Yt)
    k <- ncol(Z)
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

    for (isim in 1:(nsim + burnin)) {
        # sample beta
        Kbeta <- as.matrix(iVbeta + kronecker(iSig, ZZ))
        Cbeta <- chol(Kbeta)   # upper factor: Kbeta = t(Cbeta) %*% Cbeta
        rhs <- as.numeric(iVbeta %*% beta0) + as.vector(ZY %*% iSig)
        beta_hat <- backsolve(Cbeta, backsolve(Cbeta, rhs, transpose = TRUE))
        beta <- as.numeric(beta_hat + backsolve(Cbeta, rnorm(n*k)))

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
        }
    }

    # aggregate across draws
    pf <- mean(store_mu)
    log_w <- -0.5*log(2*pi*store_var) - 0.5*(yreal - store_mu)^2/store_var
    lpl <- log(mean(exp(log_w - max(log_w)))) + max(log_w)

    list(pf = pf, lpl = lpl)
}
