# pred_largeVAR_FSV.R
# This function computes the one-step-ahead posterior predictive means
# and log predictive likelihoods of the target variables from the
# large VAR with factor stochastic volatility (Section 14.4),
#     y_t = A' x_t + L*f_t + u_t,
#     u_t ~ N(0, D_t),   f_t ~ N(0, G_t),
#     D_t = diag(exp(h_{1t}), ..., exp(h_{nt})),
#     G_t = diag(exp(h_{n+1,t}), ..., exp(h_{n+r,t})),
# so that the error covariance matrix is Sig_t = L*G_t*L' + D_t. The
# loading matrix L is left unrestricted; the factors and loadings are
# then not separately identified, but Sig_t is, and it is all the
# predictive density requires. The r factor log-volatilities follow
# zero-mean stationary AR(1) processes (their scales are absorbed into
# the loadings), while the n idiosyncratic log-volatilities follow
# stationary AR(1) processes with free levels mu_i.
#
# The priors on the VAR coefficients ('minn', 'hs', 'mahp') are as
# in pred_largeVAR_SV.R; each free loading has the prior N(0, 1).
# Requires Minn_indep.R, SVAR1.R and sample_SVAR1para_mu.R.
#
# Inputs:
# Y:          T x n matrix of observations
# Y0:         p0 x n matrix of pre-sample observations (p0 >= p)
# Tt:         scalar; forecast origin (estimation uses Y[1:Tt,], the
#             forecast target is Y[Tt+1,])
# p:          scalar; lag order
# r:          scalar; number of factors
# targets:    length-q vector of indices of the target variables
# prior_type: 'minn', 'hs', or 'mahp'
# kappa1:     scalar; intercept prior variance
# kappa2:     scalar; own-lag tightness ('minn') or initial value
# kappa3:     scalar; cross-lag tightness ('minn') or initial value
# nsim:       scalar; number of post-burnin MCMC draws
# burnin:     scalar; number of burnin draws
#
# Outputs (returned as a list):
# pf:        length-q posterior predictive means of the targets
# lpl_joint: scalar; log of the joint predictive density of the q
#            targets at their realized values (log-mean-exp over draws)
# lpl:       length-q marginal log predictive likelihoods of the targets

suppressMessages(library(Matrix))

pred_largeVAR_FSV <- function(Y, Y0, Tt, p, r, targets, prior_type, kappa1,
                              kappa2, kappa3, nsim, burnin) {
    n <- ncol(Y)
    k <- 1 + n*p
    q <- length(targets)
    Yt <- Y[1:Tt, , drop = FALSE]
    yreal <- Y[Tt+1, targets]

    # priors for the volatility parameters; the factor log-volatilities
    # (last r elements) have their levels fixed at zero
    mu0 <- numeric(n+r);            Vmu <- 100*rep(1, n+r)
    phi0 <- 0.98*rep(1, n+r);       Vphi <- 0.05^2*rep(1, n+r)
    nu_h <- 3*rep(1, n+r);          S_h <- 0.1*(nu_h - 1)
    free_mu <- c(rep(TRUE, n), rep(FALSE, r))
    V_l <- 1   # prior variance of each free loading

    # construct the regressor matrix Z = [1, y_{t-1}', ..., y_{t-p}']
    tmpY <- rbind(Y0[(nrow(Y0)-p+1):nrow(Y0), , drop = FALSE], Yt)
    Z <- matrix(0, Tt, n*p)
    for (i in 1:p) {
        Z[, ((i-1)*n+1):(i*n)] <- tmpY[(p-i+1):(nrow(tmpY)-i), , drop = FALSE]
    }
    Z <- cbind(1, Z)

    # Minnesota structure: beta0 = 0 and the constants C_ij (V_Minn with
    # unit tightness); is_own/is_cross flag own- and cross-lag positions
    mi <- Minn_indep(p, kappa1, kappa2, kappa3, Y0, Yt, 0)
    beta0 <- mi$beta_Minn; V_alpha <- mi$V_Minn
    C_Minn <- Minn_indep(p, kappa1, 1, 1, Y0, Yt, 0)$V_Minn
    is_int <- rep(FALSE, n*k);  is_own <- rep(FALSE, n*k)
    for (i in 1:n) {
        is_int[(i-1)*k + 1] <- TRUE
        for (l in 1:p) {
            is_own[(i-1)*k + 1 + (l-1)*n + i] <- TRUE
        }
    }
    is_cross <- !is_int & !is_own
    nlag <- n*k - n     # number of lag coefficients

    # regressor row for forecasting Y[Tt+1,]
    xtp1 <- c(1, as.vector(t(Yt[Tt:(Tt-p+1), , drop = FALSE])))

    # initialize the Markov chain
    A <- matrix(0, k, n)
    mu <- numeric(n+r)
    ZZ <- crossprod(Z)
    for (i in 1:n) {
        iVai <- 1/V_alpha[((i-1)*k+1):(i*k)]
        Ki <- ZZ; diag(Ki) <- diag(Ki) + iVai
        A[, i] <- solve(Ki, crossprod(Z, Yt[, i]))
        mu[i] <- log(mean((Yt[, i] - as.numeric(Z %*% A[, i]))^2)/2)
    }
    h <- cbind(matrix(mu[1:n], Tt, n, byrow = TRUE), matrix(0, Tt, r))
    phi <- phi0
    sigh2 <- 0.05*rep(1, n+r)
    L <- rbind(diag(r), 0.1*matrix(1, n-r, r))
    F <- matrix(0, Tt, r)
        # global-local prior states
    tau2 <- rep(1, n*k);  nu_tau <- rep(1, n*k)
    theta2 <- 0.01;  xi_theta <- 1
    xi_k2 <- 1;  xi_k3 <- 1

    store_lden <- matrix(0, nsim, q+1)
    store_mu <- matrix(0, nsim, q)
    for (isim in 1:(nsim + burnin)) {
        # update the prior covariance of beta under the global-local priors
        if (prior_type == "hs") {
            V_alpha[!is_int] <- theta2*tau2[!is_int]
        } else if (prior_type == "mahp") {
            V_alpha[is_own] <- kappa2*tau2[is_own]*C_Minn[is_own]
            V_alpha[is_cross] <- kappa3*tau2[is_cross]*C_Minn[is_cross]
        }

        # sample the factors f_1, ..., f_T jointly via the precision sampler
        e <- as.vector(t(Yt - Z %*% A))
        Xf <- kronecker(Diagonal(Tt), Matrix(L, sparse = TRUE))
        XfiSig <- t(Xf) %*% Diagonal(x = as.vector(t(exp(-h[, 1:n, drop = FALSE]))))
        Kf <- Diagonal(x = as.vector(t(exp(-h[, (n+1):(n+r), drop = FALSE])))) +
              XfiSig %*% Xf
        Kf <- forceSymmetric(Kf)
        CKf <- chol(Kf)                   # UPPER sparse factor
        f_hat <- solve(CKf, solve(t(CKf), XfiSig %*% e))
        f <- as.numeric(f_hat + solve(CKf, rnorm(Tt*r)))
        F <- t(matrix(f, r, Tt))

        # sample (beta_i, l_i) equation by equation: given the factors, the
        # n equations are independent Gaussian regressions
        Zi <- cbind(Z, F)
        for (i in 1:n) {
            wi <- exp(-h[, i])
            iVti <- c(1/V_alpha[((i-1)*k+1):(i*k)], rep(1, r)/V_l)
            ti0 <- c(beta0[((i-1)*k+1):(i*k)], numeric(r))
            Kti <- crossprod(Zi, wi*Zi)
            diag(Kti) <- diag(Kti) + iVti
            # upper Cholesky factor; backsolve(., transpose = TRUE) is the
            # MATLAB lower-triangular solve CKti \ x
            CKti <- chol(Kti)
            ti_hat <- backsolve(CKti, backsolve(CKti,
                iVti*ti0 + crossprod(Zi, wi*Yt[, i]), transpose = TRUE))
            ti <- as.numeric(ti_hat + backsolve(CKti, rnorm(k+r)))
            A[, i] <- ti[1:k]
            L[i, ] <- ti[(k+1):(k+r)]
        }

        # sample the log-volatilities series by series
        U <- Yt - Z %*% A - F %*% t(L)
        ystar <- log(cbind(U, F)^2 + 1e-4)
        for (i in 1:(n+r)) {
            h[, i] <- SVAR1(ystar[, i], h[, i], mu[i], phi[i], sigh2[i])
        }

        # sample the volatility parameters (mu, phi, sigh2)
        vp <- sample_SVAR1para_mu(h, mu, phi, sigh2, mu0, Vmu, phi0, Vphi,
                                  nu_h, S_h, free_mu)
        mu <- vp$mu; phi <- vp$phi; sigh2 <- vp$sigh2

        # sample the global-local prior hyperparameters
        # R's rgamma takes shape and scale
        alp <- as.vector(A)
        if (prior_type == "hs") {
            b2 <- alp[!is_int]^2
            tau2[!is_int] <- 1/rgamma(nlag, shape = 1,
                scale = 1/(1/nu_tau[!is_int] + b2/(2*theta2)))
            nu_tau[!is_int] <- 1/rgamma(nlag, shape = 1,
                scale = 1/(1 + 1/tau2[!is_int]))
            theta2 <- 1/rgamma(1, shape = (nlag+1)/2,
                scale = 1/(1/xi_theta + sum(b2/tau2[!is_int])/2))
            xi_theta <- 1/rgamma(1, shape = 1, scale = 1/(1 + 1/theta2))
        } else if (prior_type == "mahp") {
            kap <- kappa2*is_own + kappa3*is_cross
            b2 <- alp[!is_int]^2
            denom <- kap[!is_int]*C_Minn[!is_int]
            tau2[!is_int] <- 1/rgamma(nlag, shape = 1,
                scale = 1/(1/nu_tau[!is_int] + b2/(2*denom)))
            nu_tau[!is_int] <- 1/rgamma(nlag, shape = 1,
                scale = 1/(1 + 1/tau2[!is_int]))
            so <- sum(alp[is_own]^2/(tau2[is_own]*C_Minn[is_own]))
            sc <- sum(alp[is_cross]^2/(tau2[is_cross]*C_Minn[is_cross]))
            kappa2 <- 1/rgamma(1, shape = (n*p+1)/2, scale = 1/(1/xi_k2 + so/2))
            xi_k2 <- 1/rgamma(1, shape = 1, scale = 1/(1 + 1/kappa2))
            kappa3 <- 1/rgamma(1, shape = (n*(n-1)*p+1)/2,
                               scale = 1/(1/xi_k3 + sc/2))
            xi_k3 <- 1/rgamma(1, shape = 1, scale = 1/(1 + 1/kappa3))
        }

        if (isim > burnin) {
            isave <- isim - burnin
            # forecast h_{Tt+1} via the AR(1) and the implied Sig_{Tt+1}
            htp1 <- mu + phi*(h[Tt, ] - mu) + sqrt(sigh2)*rnorm(n+r)
            mu_full <- as.numeric(xtp1 %*% A)
            mu_q <- mu_full[targets]
            Sig_tp1 <- diag(exp(htp1[1:n]), nrow = n) +
                       L %*% diag(exp(htp1[(n+1):(n+r)]), nrow = r) %*% t(L)
            Sig_q <- Sig_tp1[targets, targets, drop = FALSE]
            Sig_q <- (Sig_q + t(Sig_q))/2
            CSig_q <- chol(Sig_q)          # UPPER factor
            u <- backsolve(CSig_q, yreal - mu_q, transpose = TRUE)
            lden_joint <- -q/2*log(2*pi) - sum(log(diag(CSig_q))) - 0.5*sum(u^2)
            dSig_q <- diag(Sig_q)
            lden <- -0.5*log(2*pi*dSig_q) - 0.5*(yreal - mu_q)^2/dSig_q
            store_mu[isave, ] <- mu_q
            store_lden[isave, ] <- c(lden, lden_joint)
        }
    }

    # aggregate across draws
    pf <- colMeans(store_mu)
    tmpmax <- apply(store_lden, 2, max)
    lpls <- log(colMeans(exp(sweep(store_lden, 2, tmpmax, "-")))) + tmpmax
    lpl <- lpls[1:q]
    lpl_joint <- lpls[q+1]

    list(pf = pf, lpl_joint = lpl_joint, lpl = lpl)
}
