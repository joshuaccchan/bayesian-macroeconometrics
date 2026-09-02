# pred_largeVAR_CSV.R
# This function computes the one-step-ahead posterior predictive means
# and log predictive likelihoods of the target variables from the VAR
# with common stochastic volatility (Section 14.2),
#     y_t = A' x_t + eps_t,   eps_t ~ N(0, exp(h_t)*Sig),
#     h_t = phi*h_{t-1} + u_t^h,   u_t^h ~ N(0, sigh2),
#     h_1 ~ N(0, sigh2/(1 - phi^2)),
# with the natural conjugate prior (A, Sig) ~ NIW(A0, VA, nu0, S0)
# elicited Minnesota-style by Minn_NCP.R.
# Requires Minn_NCP.R and sample_CSV_h_ARMH.R.
#
# Inputs:
# Y:        T x n matrix of observations
# Y0:       p0 x n matrix of pre-sample observations (p0 >= p)
# Tt:       scalar; forecast origin (estimation uses Y[1:Tt,], the
#           forecast target is Y[Tt+1,])
# p:        scalar; lag order
# targets:  length-q vector of indices of the target variables
# kappa1:   scalar; prior variance on intercepts
# kappa2:   scalar; overall shrinkage on lag coefficients
# nsim:     scalar; number of post-burnin MCMC draws
# burnin:   scalar; number of burnin draws
#
# Outputs (returned as a list):
# pf:        length-q posterior predictive means of the targets
# lpl_joint: scalar; log of the joint predictive density of the q
#            targets at their realized values (log-mean-exp over draws)
# lpl:       length-q marginal log predictive likelihoods of the targets

suppressMessages(library(Matrix))

pred_largeVAR_CSV <- function(Y, Y0, Tt, p, targets, kappa1, kappa2,
                              nsim, burnin) {
    n <- ncol(Y)
    k <- 1 + n*p
    q <- length(targets)
    Yt <- Y[1:Tt, , drop = FALSE]
    yreal <- Y[Tt+1, targets]

    # MATLAB's iwishrnd (Statistics Toolbox) has no base-R counterpart: draw
    # Sig ~ IW(S, df) from the Bartlett decomposition of a Wishart(df, S^{-1}).
    iwishrnd <- function(S, df) {
        nn <- nrow(S)
        C <- chol(S)                     # upper: S = t(C) %*% C
        Abar <- matrix(0, nn, nn)        # lower-triangular Bartlett factor
        diag(Abar) <- sqrt(rchisq(nn, df - (1:nn) + 1))
        Abar[lower.tri(Abar)] <- rnorm(nn*(nn-1)/2)
        crossprod(forwardsolve(Abar, C))
    }

    # priors for the volatility parameters
    phi0 <- 0.98; Vphi <- 0.05^2       # phi ~ TN_(-1,1)(phi0, Vphi)
    nu_h <- 3; S_h <- 0.1*(nu_h - 1)   # sigh2 ~ IG(nu_h, S_h)

    # construct the regressor matrix Z = [1, y_{t-1}', ..., y_{t-p}']
    tmpY <- rbind(Y0[(nrow(Y0)-p+1):nrow(Y0), , drop = FALSE], Yt)
    Z <- matrix(0, Tt, n*p)
    for (i in 1:p) {
        Z[, ((i-1)*n+1):(i*n)] <- tmpY[(p-i+1):(nrow(tmpY)-i), , drop = FALSE]
    }
    Z <- cbind(1, Z)

    # natural conjugate prior
    prior <- Minn_NCP(Yt, Y0, p, kappa1, kappa2, 0)
    A0 <- prior$A0; VA <- prior$VA; nu0 <- prior$nu0; S0 <- prior$S0
    iva <- 1/diag(VA)          # diagonal of VA^{-1} (sparse iVA in the .m)

    # regressor row for forecasting Y[Tt+1,]
    xtp1 <- c(1, as.vector(t(Yt[Tt:(Tt-p+1), , drop = FALSE])))

    # initialize the Markov chain
    phi <- phi0
    sigh2 <- 0.05
    h <- numeric(Tt)

    store_lden <- matrix(0, nsim, q+1)
    store_mu <- matrix(0, nsim, q)
    for (isim in 1:(nsim + burnin)) {
        # sample Sig and A given h with Omega^{-1} = diag(exp(-h))
        iOm <- exp(-h)
        ZiOm <- t(Z*iOm)                   # k x Tt, = Z' * diag(iOm)
        KA <- ZiOm %*% Z
        diag(KA) <- diag(KA) + iva
        # CKA is the UPPER factor (MATLAB stores the lower one);
        # backsolve(CKA, x, transpose = TRUE) is MATLAB's CKA_lower \ x
        CKA <- chol(KA)
        A_hat <- backsolve(CKA, backsolve(CKA, iva*A0 + ZiOm %*% Yt,
                                          transpose = TRUE))
        S_hat <- S0 + crossprod(A0, iva*A0) + crossprod(Yt, iOm*Yt) -
                 crossprod(A_hat, KA %*% A_hat)
        S_hat <- (S_hat + t(S_hat))/2      # adjust for rounding errors
        Sig <- iwishrnd(S_hat, nu0 + Tt)
        CSig <- t(chol(Sig))               # lower factor
        A <- A_hat + backsolve(CKA, matrix(rnorm(k*n), k, n)) %*% t(CSig)

        # sample the common log-volatility h via the Laplace-based ARMH step
        U <- Yt - Z %*% A
        tmp <- t(forwardsolve(CSig, t(U)))  # residuals standardized by chol(Sig)
        s2 <- rowSums(tmp^2)
        h <- sample_CSV_h_ARMH(s2, phi, sigh2, h, n, 30)$h

        # sample sigh2
        eh <- c(h[1]*sqrt(1-phi^2), h[2:Tt] - phi*h[1:(Tt-1)])
        # R's rgamma takes shape and scale
        sigh2 <- 1/rgamma(1, shape = nu_h + Tt/2, scale = 1/(S_h + sum(eh^2)/2))

        # sample phi via an independence-chain MH step
        Kphi <- 1/Vphi + sum(h[1:(Tt-1)]^2)/sigh2
        phihat <- (phi0/Vphi + sum(h[1:(Tt-1)]*h[2:Tt])/sigh2)/Kphi
        phic <- phihat + rnorm(1)/sqrt(Kphi)
        gphi <- function(x) 0.5*log(1-x^2) - 0.5*(1-x^2)/sigh2*h[1]^2
        if (abs(phic) < 0.998 && exp(gphi(phic) - gphi(phi)) > runif(1)) {
            phi <- phic
        }

        if (isim > burnin) {
            isave <- isim - burnin
            # forecast h_{Tt+1} via the AR(1) and the implied Sig_{Tt+1}
            htp1 <- phi*h[Tt] + sqrt(sigh2)*rnorm(1)
            mu_full <- as.numeric(xtp1 %*% A)
            mu_q <- mu_full[targets]
            Sig_q <- exp(htp1)*Sig[targets, targets, drop = FALSE]
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
