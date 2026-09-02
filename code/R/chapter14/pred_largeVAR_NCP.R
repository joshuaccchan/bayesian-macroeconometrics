# pred_largeVAR_NCP.R
# This function computes the one-step-ahead posterior predictive means
# and log predictive likelihoods of the target variables from the
# homoskedastic VAR
#     y_t = A' x_t + eps_t,   eps_t ~ N(0, Sig),
# with the natural conjugate prior (A, Sig) ~ NIW(A0, VA, nu0, S0)
# elicited Minnesota-style by Minn_NCP.R. The NIW posterior is
# available in closed form, so each iteration draws (A, Sig) directly
# from the posterior and evaluates the predictive density; the log
# predictive likelihoods are then formed by averaging over the draws.
# Requires Minn_NCP.R.
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
# nsim:     scalar; number of posterior draws
#
# Outputs (returned as a list):
# pf:        length-q posterior predictive means of the targets
# lpl_joint: scalar; log of the joint predictive density of the q
#            targets at their realized values (log-mean-exp over draws)
# lpl:       length-q marginal log predictive likelihoods of the targets

pred_largeVAR_NCP <- function(Y, Y0, Tt, p, targets, kappa1, kappa2, nsim) {
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

    # construct the regressor matrix Z = [1, y_{t-1}', ..., y_{t-p}']
    tmpY <- rbind(Y0[(nrow(Y0)-p+1):nrow(Y0), , drop = FALSE], Yt)
    Z <- matrix(0, Tt, n*p)
    for (i in 1:p) {
        Z[, ((i-1)*n+1):(i*n)] <- tmpY[(p-i+1):(nrow(tmpY)-i), , drop = FALSE]
    }
    Z <- cbind(1, Z)

    # natural conjugate prior and analytical posterior
    prior <- Minn_NCP(Yt, Y0, p, kappa1, kappa2, 0)
    A0 <- prior$A0; VA <- prior$VA; nu0 <- prior$nu0; S0 <- prior$S0
    iva <- 1/diag(VA)             # diagonal of VA^{-1} (sparse iVA in the .m)
    KA <- crossprod(Z)
    diag(KA) <- diag(KA) + iva
    # CKA is the UPPER factor (MATLAB stores the lower one): KA = t(CKA)%*%CKA;
    # backsolve(CKA, x, transpose = TRUE) is MATLAB's CKA_lower \ x
    CKA <- chol(KA)
    A_hat <- backsolve(CKA, backsolve(CKA, iva*A0 + crossprod(Z, Yt),
                                      transpose = TRUE))
    S_hat <- S0 + crossprod(A0, iva*A0) + crossprod(Yt) -
             crossprod(A_hat, KA %*% A_hat)
    S_hat <- (S_hat + t(S_hat))/2    # adjust for rounding errors
    nu_hat <- nu0 + Tt

    # regressor row for forecasting Y[Tt+1,]
    xtp1 <- c(1, as.vector(t(Yt[Tt:(Tt-p+1), , drop = FALSE])))

    store_lden <- matrix(0, nsim, q+1)
    store_mu <- matrix(0, nsim, q)
    for (isim in 1:nsim) {
        # sample Sig and A from the NIW posterior
        Sig <- iwishrnd(S_hat, nu_hat)
        CSig <- t(chol(Sig))                # lower factor
        A <- A_hat + backsolve(CKA, matrix(rnorm(k*n), k, n)) %*% t(CSig)

        # evaluate the predictive density at the realized targets
        mu_full <- as.numeric(xtp1 %*% A)
        mu_q <- mu_full[targets]
        Sig_q <- Sig[targets, targets, drop = FALSE]
        CSig_q <- chol(Sig_q)               # UPPER factor
        u <- backsolve(CSig_q, yreal - mu_q, transpose = TRUE)
        lden_joint <- -q/2*log(2*pi) - sum(log(diag(CSig_q))) - 0.5*sum(u^2)
        dSig_q <- diag(Sig_q)
        lden <- -0.5*log(2*pi*dSig_q) - 0.5*(yreal - mu_q)^2/dSig_q
        store_mu[isim, ] <- mu_q
        store_lden[isim, ] <- c(lden, lden_joint)
    }

    # aggregate across draws
    pf <- colMeans(store_mu)
    tmpmax <- apply(store_lden, 2, max)
    lpls <- log(colMeans(exp(sweep(store_lden, 2, tmpmax, "-")))) + tmpmax
    lpl <- lpls[1:q]
    lpl_joint <- lpls[q+1]

    list(pf = pf, lpl_joint = lpl_joint, lpl = lpl)
}
