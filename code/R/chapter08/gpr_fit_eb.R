# gpr_fit_eb.R
# Empirical-Bayes hyperparameters for Gaussian process regression with an
# ARD squared exponential kernel. The model is
#   y_t = f(x_t) + eps_t,  eps_t ~ N(0, sig2),  f ~ GP(0, k),
#   k(x,x') = sigf2 * exp(-0.5 * sum_j (x_j - x'_j)^2 / lam_j^2).
# The hyperparameters (sigf2, lam_1..lam_d, sig2) are set to the maximizers
# of the log integrated likelihood, found by optimizing on the log scale (so
# positivity is automatic) from a median-heuristic start with several
# restarts. The objective is the local function gp_se_negloglike below,
# minimized with optim's Nelder-Mead (the counterpart of MATLAB's fminsearch).
#
# Inputs:
#   y    : length-n response
#   X    : n-by-d input matrix
#   opts : (optional) list with fields $nstart (restarts, default 5) and
#          $verbose (print per-restart progress, default FALSE)
#
# Outputs (returned as a list):
#   hp         : list with fields sigf2 (scalar), lam (length-d), sig2 (scalar)
#   logintlike : maximized log integrated likelihood
#   info       : list with fields logp (optimizer solution) and Dc
#                (per-dimension squared-distance list)

gpr_fit_eb <- function(y, X, opts = list()) {
    if (is.null(opts$nstart))  opts$nstart  <- 5
    if (is.null(opts$verbose)) opts$verbose <- FALSE

    n <- nrow(X)
    d <- ncol(X)

    # per-dimension squared distances; median-heuristic starting length scales
    Dc   <- vector("list", d)
    lam0 <- numeric(d)
    for (j in 1:d) {
        Dj      <- outer(X[, j], X[, j], "-")^2
        Dc[[j]] <- Dj
        dv      <- sqrt(Dj[upper.tri(Dj)])   # strictly upper triangle
        lam0[j] <- median(dv[dv > 0])
        if (!isTRUE(lam0[j] > 0)) lam0[j] <- 1
    }

    vy    <- var(y)
    logp0 <- c(log(vy), log(lam0), log(vy/4))   # starting point, log scale
    obj   <- function(logp) gp_se_negloglike(logp, y, Dc)  # neg. log int. lik.
    # optim's Nelder-Mead takes maxit as the function-evaluation budget and a
    # relative function tolerance; reltol = 1e-8 stands in for MATLAB's
    # TolX = TolFun = 1e-6 (the objective is of order 1e2 here)
    optfs <- list(maxit = 4000, reltol = 1e-8)

    # optimize from the median-heuristic start plus perturbed restarts
    best_logp <- logp0
    best_nll  <- Inf
    for (s in 1:opts$nstart) {
        start <- logp0
        if (s > 1) start <- logp0 + 0.7*rnorm(d + 2)   # perturb on log scale
        res <- optim(start, obj, method = "Nelder-Mead", control = optfs)
        lp  <- res$par
        nll <- res$value
        if (nll < best_nll) {
            best_nll  <- nll
            best_logp <- lp
        }
        if (opts$verbose) {
            cat(sprintf("  restart %d: log int. lik. = %.4f\n", s, -nll))
        }
    }

    # unpack hyperparameters
    hp <- list(sigf2 = exp(best_logp[1]),
               lam   = exp(best_logp[2:(d+1)]),   # length-d length scales
               sig2  = exp(best_logp[d+2]))
    logintlike <- -best_nll
    info <- list(logp = best_logp, Dc = Dc)
    list(hp = hp, logintlike = logintlike, info = info)
}


gp_se_negloglike <- function(logp, y, Dc) {
    # Negative log integrated likelihood of the ARD squared exponential GP at
    # the log-hyperparameters logp = [log(sigf2), log(lam_1..lam_d), log(sig2)].
    # Integrating f out gives (y | theta) ~ N(0, Ksig) with Ksig = K + sig2*I,
    # so the likelihood is evaluated through the Cholesky factor of Ksig.
    # Returns a large value if Ksig stays numerically indefinite (so the
    # optimizer backs off).
    #
    # Inputs:
    #   logp : log-hyperparameters [log(sigf2), log(lam_1..lam_d), log(sig2)]
    #   y    : length-n response
    #   Dc   : list of d per-dimension squared-distance matrices (each n-by-n)
    #
    # Output:
    #   nll  : negative log integrated likelihood at logp

    n <- length(y)
    d <- length(Dc)
    sigf2 <- exp(logp[1])
    lam   <- exp(logp[2:(d+1)])
    sig2  <- exp(logp[d+2])

    M <- matrix(0, n, n)   # scaled squared distances
    for (j in 1:d) {
        M <- M + Dc[[j]]/(lam[j]^2)
    }
    Ksig <- sigf2*exp(-0.5*M) + sig2*diag(n)

    # R's chol() is the UPPER factor (Ksig = t(C) %*% C) and errors out where
    # MATLAB's chol returns the flag p > 0, so wrap it in tryCatch
    C <- tryCatch(chol(Ksig), error = function(e) NULL)
    if (is.null(C)) {        # escalate jitter if not pos. def.
        jit <- 1e-8*(sum(diag(Ksig))/n + 1)
        for (k in 1:6) {
            C <- tryCatch(chol(Ksig + jit*diag(n)), error = function(e) NULL)
            if (!is.null(C)) break
            jit <- jit*10
        }
        if (is.null(C)) return(1e10)
    }

    a <- backsolve(C, y, transpose = TRUE)   # a'a = y' Ksig^{-1} y
    logdet <- 2*sum(log(diag(C)))            # log|Ksig|
    nll <- 0.5*sum(a*a) + 0.5*logdet + 0.5*n*log(2*pi)
    nll
}
