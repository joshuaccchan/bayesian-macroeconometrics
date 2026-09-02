# pred_largeVAR_OISV.R
# This function computes the one-step-ahead posterior predictive means
# and log predictive likelihoods of the target variables from the
# large VAR with order-invariant stochastic volatility (Section 14.3),
#     y_t = A' x_t + eps_t,    eps_t ~ N(0, Sig_t),
#     Sig_t^{-1} = B0' D_t^{-1} B0,
#     D_t = diag(exp(h_{1t}), ..., exp(h_{nt})),
# where B0 is an unrestricted (dense) n x n matrix and each
# log-volatility follows the zero-mean stationary AR(1) process
#     h_{it} = phi_i*h_{i,t-1} + u_{it}^h,   u_{it}^h ~ N(0, sigh2_i),
#     h_{i1} ~ N(0, sigh2_i/(1 - phi_i^2)).
#
# The priors on the VAR coefficients ('minn', 'hs', 'mahp') are as in
# pred_largeVAR_SV.R; each element of B0 has the prior N(b0_ij, 10)
# with prior mean the identity matrix.
# Requires Minn_indep.R, SVAR1.R, sample_SVAR1para.R and sample_B0.R.
#
# Inputs:
# Y:          T x n matrix of observations
# Y0:         p0 x n matrix of pre-sample observations (p0 >= p)
# Tt:         scalar; forecast origin (estimation uses Y[1:Tt,], the
#             forecast target is Y[Tt+1,])
# p:          scalar; lag order
# targets:    length-q vector of indices of the target variables
# prior_type: 'minn', 'minnH', 'hs', or 'mahp' (see pred_largeVAR_SV.R)
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
# hyp:       nsim x 3 saved hyperparameter draws [kappa2, kappa3,
#            theta2] (diagnostics): kappa2/kappa3 are the sampled
#            tightness under 'minnH'/'mahp', theta2 the horseshoe global
#            scale under 'hs'; unused entries stay at their input values
# draws_mu:  nsim x q predictive-mean draws, for comparing the mean and
#            median point forecast

pred_largeVAR_OISV <- function(Y, Y0, Tt, p, targets, prior_type, kappa1,
                               kappa2, kappa3, nsim, burnin) {
    n <- ncol(Y)
    k <- 1 + n*p
    q <- length(targets)
    Yt <- Y[1:Tt, , drop = FALSE]
    yreal <- Y[Tt+1, targets]

    # priors for the volatility parameters and B0
    phi0 <- 0.95;                 Vphi <- 0.05^2
    nu_h <- 3*rep(1, n);          S_h <- 0.05*(nu_h - 1)
    b0_B0 <- diag(n);             iV_B0 <- diag(n)/10

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
    ZZ <- crossprod(Z)
    for (i in 1:n) {
        iVai <- 1/V_alpha[((i-1)*k+1):(i*k)]
        Ki <- ZZ; diag(Ki) <- diag(Ki) + iVai
        A[, i] <- solve(Ki, crossprod(Z, Yt[, i]))
    }
    U <- Yt - Z %*% A
    Sig_hat <- crossprod(U)/Tt
    B0 <- diag(1/sqrt(diag(Sig_hat)))
    h <- matrix(0, Tt, n)
    phi <- phi0*rep(1, n)
    sigh2 <- 0.05*rep(1, n)
        # global-local prior states
    tau2 <- rep(1, n*k);  nu_tau <- rep(1, n*k)
    theta2 <- 0.01;  xi_theta <- 1
    xi_k2 <- 1;  xi_k3 <- 1

    store_lden <- matrix(0, nsim, q+1)
    store_mu <- matrix(0, nsim, q)
    store_hyp <- matrix(0, nsim, 3)   # [kappa2, kappa3, theta2] per saved draw
    for (isim in 1:(nsim + burnin)) {
        # update the prior covariance of beta under the global-local priors
        if (prior_type == "hs") {
            V_alpha[!is_int] <- theta2*tau2[!is_int]
        } else if (prior_type == "minnH") {
            V_alpha[is_own] <- kappa2*C_Minn[is_own]
            V_alpha[is_cross] <- kappa3*C_Minn[is_cross]
        } else if (prior_type == "mahp") {
            V_alpha[is_own] <- kappa2*tau2[is_own]*C_Minn[is_own]
            V_alpha[is_cross] <- kappa3*tau2[is_cross]*C_Minn[is_cross]
        }

        # sample B0 row by row via the Waggoner-Zha-Villani scheme
        U <- Yt - Z %*% A
        B0 <- sample_B0(B0, U, h, b0_B0, iV_B0)

        # sample beta equation by equation (Corollary 14.1)
        eh_inv <- exp(-h)
        for (i in 1:n) {
            A[, i] <- 0
            Etil <- (Yt - Z %*% A) %*% t(B0) # row t is z_t' = (B0(y_t - A_{-i}'x_t))'
            bi <- B0[, i]
            w <- as.numeric(eh_inv %*% bi^2) # w_it = sum_j B0(j,i)^2 exp(-h_jt)
            cv <- as.numeric((eh_inv*Etil) %*% bi)   # c in the .m
            iVai <- 1/V_alpha[((i-1)*k+1):(i*k)]
            Kai <- crossprod(Z, w*Z)
            diag(Kai) <- diag(Kai) + iVai
            # upper Cholesky factor; backsolve(., transpose = TRUE) is the
            # MATLAB lower-triangular solve CKai \ x
            CKai <- chol(Kai)
            ai_hat <- backsolve(CKai, backsolve(CKai,
                iVai*beta0[((i-1)*k+1):(i*k)] + crossprod(Z, cv),
                transpose = TRUE))
            A[, i] <- ai_hat + backsolve(CKai, rnorm(k))
        }

        # sample the log-volatilities equation by equation; each h_i is
        # recentered to mean zero, since its level is absorbed into the
        # scale of the corresponding row of B0
        Eorth <- (Yt - Z %*% A) %*% t(B0)
        ystar <- log(Eorth^2 + 1e-4)
        for (i in 1:n) {
            h[, i] <- SVAR1(ystar[, i], h[, i], 0, phi[i], sigh2[i])
            h[, i] <- h[, i] - mean(h[, i])
        }

        # sample the volatility parameters (phi, sigh2)
        vp <- sample_SVAR1para(h, phi, sigh2, phi0, Vphi, nu_h, S_h)
        phi <- vp$phi; sigh2 <- vp$sigh2

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
        } else if (prior_type == "minnH") {
            so <- sum(alp[is_own]^2/C_Minn[is_own])
            sc <- sum(alp[is_cross]^2/C_Minn[is_cross])
            kappa2 <- 1/rgamma(1, shape = (n*p+1)/2, scale = 1/(1/xi_k2 + so/2))
            xi_k2 <- 1/rgamma(1, shape = 1, scale = 1/(1 + 1/kappa2))
            kappa3 <- 1/rgamma(1, shape = (n*(n-1)*p+1)/2,
                               scale = 1/(1/xi_k3 + sc/2))
            xi_k3 <- 1/rgamma(1, shape = 1, scale = 1/(1 + 1/kappa3))
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
            htp1 <- phi*h[Tt, ] + sqrt(sigh2)*rnorm(n)
            mu_full <- as.numeric(xtp1 %*% A)
            mu_q <- mu_full[targets]
            Sig_tp1 <- t(solve(B0, t(solve(B0, diag(exp(htp1), nrow = n)))))
            Sig_q <- Sig_tp1[targets, targets, drop = FALSE]
            Sig_q <- (Sig_q + t(Sig_q))/2
            CSig_q <- chol(Sig_q)          # UPPER factor
            u <- backsolve(CSig_q, yreal - mu_q, transpose = TRUE)
            lden_joint <- -q/2*log(2*pi) - sum(log(diag(CSig_q))) - 0.5*sum(u^2)
            dSig_q <- diag(Sig_q)
            lden <- -0.5*log(2*pi*dSig_q) - 0.5*(yreal - mu_q)^2/dSig_q
            store_mu[isave, ] <- mu_q
            store_lden[isave, ] <- c(lden, lden_joint)
            store_hyp[isave, ] <- c(kappa2, kappa3, theta2)
        }
    }

    # aggregate across draws
    pf <- colMeans(store_mu)
    tmpmax <- apply(store_lden, 2, max)
    lpls <- log(colMeans(exp(sweep(store_lden, 2, tmpmax, "-")))) + tmpmax
    lpl <- lpls[1:q]
    lpl_joint <- lpls[q+1]
    hyp <- store_hyp        # saved [kappa2 kappa3 theta2] draws (diagnostics)
    draws_mu <- store_mu    # nsim x q predictive-mean draws (mean vs median)

    list(pf = pf, lpl_joint = lpl_joint, lpl = lpl, hyp = hyp,
         draws_mu = draws_mu)
}
