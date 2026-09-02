# gpr_predict.R
# Posterior predictive moments for Gaussian process regression with an ARD
# squared exponential kernel, at test inputs Xstar given hyperparameters hp:
#   E[f(x_{T+1}) | y]   = k_{T+1}' * Ksig^{-1} * y,
#   Var[f(x_{T+1}) | y] = k_{T+1,T+1} - k_{T+1}' * Ksig^{-1} * k_{T+1},
# where Ksig = K + sig2*I and k_{T+1} is the cross-covariance between a test
# input and the T training inputs. Everything goes through the Cholesky factor
# of Ksig.
#
# Inputs:
#   y     : length-T response
#   X     : T-by-d training inputs
#   Xstar : m-by-d test inputs
#   hp    : list with fields sigf2 (scalar), lam (length-d), sig2 (scalar)
#
# Outputs (returned as a list):
#   mu  : length-m posterior mean of f at Xstar
#   s2f : length-m posterior variance of f at Xstar
#   s2y : length-m predictive variance of a noisy observation (= s2f + sig2)

gpr_predict <- function(y, X, Xstar, hp) {
    T <- nrow(X)
    d <- ncol(X)
    m <- nrow(Xstar)
    sigf2 <- hp$sigf2
    lam   <- as.numeric(hp$lam)
    sig2  <- hp$sig2

    # ARD scaled squared distances: training-training (M) and training-test (Ms)
    M  <- matrix(0, T, T)
    Ms <- matrix(0, T, m)
    for (j in 1:d) {
        xj <- X[, j]
        M  <- M  + (outer(xj, xj, "-")^2)/(lam[j]^2)
        Ms <- Ms + (outer(xj, Xstar[, j], "-")^2)/(lam[j]^2)
    }
    Ksig <- sigf2*exp(-0.5*M) + sig2*diag(T)   # Ksig = K + sig2 I
    Ks   <- sigf2*exp(-0.5*Ms)                 # cross-covariances k_{T+1}
    C    <- chol(Ksig)   # UPPER factor: Ksig = t(C) %*% C

    # Ksig^{-1} y, i.e. MATLAB's C'\(C\y) with C lower
    alpha <- backsolve(C, backsolve(C, y, transpose = TRUE))
    mu    <- as.numeric(crossprod(Ks, alpha))  # posterior mean k_{T+1}' Ksig^{-1} y

    v   <- backsolve(C, Ks, transpose = TRUE)
    # k_{T+1,T+1} - k_{T+1}' Ksig^{-1} k_{T+1}
    s2f <- pmax(sigf2 - colSums(v^2), 0)
    s2y <- s2f + sig2                          # add noise variance for y
    list(mu = mu, s2f = s2f, s2y = s2y)
}
