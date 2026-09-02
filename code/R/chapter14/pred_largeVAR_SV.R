# pred_largeVAR_SV.R
# This function computes the one-step-ahead posterior predictive means
# and log predictive likelihoods of the target variables from the
# large VAR with Cholesky stochastic volatility (Section 14.3),
#     y_t = A' x_t + eps_t,    eps_t ~ N(0, Sig_t),
#     Sig_t^{-1} = B0' D_t^{-1} B0,
#     D_t = diag(exp(h_{1t}), ..., exp(h_{nt})),
# where B0 is unit lower triangular and each log-volatility follows a
# stationary AR(1) with level mu_i,
#     h_{it} = mu_i + phi_i*(h_{i,t-1} - mu_i) + u_{it}^h,
#     u_{it}^h ~ N(0, sigh2_i),
#     h_{i1} ~ N(mu_i, sigh2_i/(1 - phi_i^2)).
#
# Four priors on the VAR coefficients are supported (Section 14.3.1);
# in each case the intercepts have fixed variance kappa1:
#   'minn':  independent Minnesota prior with own-lag tightness kappa2
#            and cross-lag tightness kappa3, both fixed;
#   'minnH': Minnesota prior with the same structure, but with the own-
#            and cross-lag tightness estimated, beta_ij ~ N(0,
#            kappa_ij*C_ij) with sqrt(kappa2), sqrt(kappa3) ~ C+(0,1)
#            and C_ij the Minnesota constants;
#   'hs':    horseshoe prior, beta_ij ~ N(0, theta^2*tau_ij^2) with
#            theta, tau_ij ~ C+(0,1);
#   'mahp':  Minnesota-type global-local prior of Chan (2021),
#            beta_ij ~ N(0, kappa_ij*tau_ij^2*C_ij) with
#            sqrt(kappa2), sqrt(kappa3), tau_ij ~ C+(0,1). It adds the
#            local scales tau_ij to 'minnH'.
# For 'minnH' and 'mahp' the inputs kappa2 and kappa3 are used as
# initial values. The half-Cauchy scales are updated with the auxiliary
# inverse-gamma representation of Makalic and Schmidt (2016).
# Requires Minn_indep.R, SVAR1.R and sample_SVAR1para_mu.R.
#
# Inputs:
# Y:          T x n matrix of observations
# Y0:         p0 x n matrix of pre-sample observations (p0 >= p)
# Tt:         scalar; forecast origin (estimation uses Y[1:Tt,], the
#             forecast target is Y[Tt+1,])
# p:          scalar; lag order
# targets:    length-q vector of indices of the target variables
# prior_type: 'minn', 'minnH', 'hs', or 'mahp'
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

pred_largeVAR_SV <- function(Y, Y0, Tt, p, targets, prior_type, kappa1,
                             kappa2, kappa3, nsim, burnin) {
    n <- ncol(Y)
    k <- 1 + n*p
    q <- length(targets)
    Yt <- Y[1:Tt, , drop = FALSE]
    yreal <- Y[Tt+1, targets]

    # priors for the volatility parameters
    mu0 <- numeric(n);            Vmu <- 100*rep(1, n)
    phi0 <- 0.98*rep(1, n);       Vphi <- 0.05^2*rep(1, n)
    nu_h <- 3*rep(1, n);          S_h <- 0.1*(nu_h - 1)

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
    beta0 <- mi$beta_Minn; V_alpha <- mi$V_Minn; s2_hat <- mi$s2_hat
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

    # prior on the free elements of B0, Minnesota-scaled:
    # B0(i,j) ~ N(0, s2_i/s2_j) for j < i
    V_b0 <- numeric(n*(n-1)/2)
    cnt <- 0
    for (i in 2:n) {
        V_b0[(cnt+1):(cnt+i-1)] <- s2_hat[i]/s2_hat[1:(i-1)]
        cnt <- cnt + i-1
    }

    # regressor row for forecasting Y[Tt+1,]
    xtp1 <- c(1, as.vector(t(Yt[Tt:(Tt-p+1), , drop = FALSE])))

    # initialize the Markov chain
    A <- matrix(0, k, n)
    mu <- numeric(n)
    ZZ <- crossprod(Z)
    for (i in 1:n) {
        iVai <- 1/V_alpha[((i-1)*k+1):(i*k)]
        Ki <- ZZ; diag(Ki) <- diag(Ki) + iVai
        A[, i] <- solve(Ki, crossprod(Z, Yt[, i]))
        mu[i] <- log(mean((Yt[, i] - as.numeric(Z %*% A[, i]))^2))
    }
    h <- matrix(mu, Tt, n, byrow = TRUE)
    phi <- phi0
    sigh2 <- 0.05*rep(1, n)
    B0 <- diag(n)
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
        } else if (prior_type == "minnH") {
            V_alpha[is_own] <- kappa2*C_Minn[is_own]
            V_alpha[is_cross] <- kappa3*C_Minn[is_cross]
        } else if (prior_type == "mahp") {
            V_alpha[is_own] <- kappa2*tau2[is_own]*C_Minn[is_own]
            V_alpha[is_cross] <- kappa3*tau2[is_cross]*C_Minn[is_cross]
        }

        # sample beta equation by equation (Corollary 14.1)
        eh_inv <- exp(-h)
        for (i in 1:n) {
            A[, i] <- 0
            Etil <- (Yt - Z %*% A) %*% t(B0)  # row t is z_t' = (B0(y_t - A_{-i}'x_t))'
            bi <- B0[, i]
            w <- as.numeric(eh_inv %*% bi^2)  # w_it = sum_j B0(j,i)^2 exp(-h_jt)
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

        # sample the free elements of B0 row by row
        E <- Yt - Z %*% A
        cnt <- 0
        for (i in 2:n) {
            Xb <- -E[, 1:(i-1), drop = FALSE]
            wb <- eh_inv[, i]
            Kb <- crossprod(Xb, wb*Xb)
            diag(Kb) <- diag(Kb) + 1/V_b0[(cnt+1):(cnt+i-1)]
            CKb <- chol(Kb)
            b_hat <- backsolve(CKb, backsolve(CKb, crossprod(Xb, wb*E[, i]),
                                              transpose = TRUE))
            B0[i, 1:(i-1)] <- b_hat + backsolve(CKb, rnorm(i-1))
            cnt <- cnt + i-1
        }

        # sample the log-volatilities equation by equation
        Eorth <- E %*% t(B0)
        ystar <- log(Eorth^2 + 1e-4)
        for (i in 1:n) {
            h[, i] <- SVAR1(ystar[, i], h[, i], mu[i], phi[i], sigh2[i])
        }

        # sample the volatility parameters (mu, phi, sigh2)
        vp <- sample_SVAR1para_mu(h, mu, phi, sigh2, mu0, Vmu, phi0, Vphi,
                                  nu_h, S_h, rep(TRUE, n))
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
            htp1 <- mu + phi*(h[Tt, ] - mu) + sqrt(sigh2)*rnorm(n)
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
