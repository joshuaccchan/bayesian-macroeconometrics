# ml_VAR_ACP.R
# Evaluates the log marginal likelihood of a VAR under the asymmetric
# conjugate prior of Chan (2022), computed equation by equation in the
# recursive structural form (with a zero coefficient prior mean).
#
# Inputs:
#   Y     : T x n matrix of observations
#   Y0    : p0 x n matrix of pre-sample observations (p0 >= p)
#   p     : lag order
#   kappa : c(kappa1, kappa2, kappa3) intercept, own-lag and other-lag shrinkage
#   s2    : n-vector of residual variances from univariate AR(p) models
#
# Output:
#   lml : log marginal likelihood

ml_VAR_ACP <- function(Y, Y0, p, kappa, s2) {
    T <- nrow(Y)
    n <- ncol(Y)

    # build the lag regressor matrix W (intercept and p lags)
    tmpY <- rbind(Y0[(nrow(Y0)-p+1):nrow(Y0), , drop = FALSE], Y)
    Z <- matrix(0, T, n*p)
    for (l in 1:p) {
        Z[, ((l-1)*n+1):(l*n)] <- tmpY[(p-l+1):(nrow(tmpY)-l), , drop = FALSE]
    }
    W <- cbind(1, Z)

    # accumulate the equation-by-equation contributions
    lml <- -T*n/2*log(2*pi)
    for (i in 1:n) {
        yi <- Y[, i]
        # regressors: contemporaneous variables and lags
        # seq_len(i-1) is empty when i = 1 (do NOT write 1:(i-1) in R)
        Xi <- cbind(-Y[, seq_len(i-1), drop = FALSE], W)

        # prior scale V_i = diag(V_alpha, V_beta), with the
        # Minnesota own- and other-lag shrinkage in V_beta
        vb <- numeric(1 + n*p)
        vb[1] <- kappa[1]
        for (l in 1:p) {
            for (r in 1:n) {
                idx <- 1 + (l-1)*n + r
                if (r == i) {
                    vb[idx] <- kappa[2]/(l^2*s2[i])   # own lag
                } else {
                    vb[idx] <- kappa[3]/(l^2*s2[r])   # other lag
                }
            }
        }
        Vi <- c(1/s2[seq_len(i-1)], vb)
        nu_i <- 1 + i/2
        S_i <- s2[i]/2

        # posterior quantities (the prior mean is zero)
        K <- diag(1/Vi) + crossprod(Xi)
        CK <- chol(K)   # UPPER factor, K = t(CK) %*% CK
        th <- backsolve(CK, forwardsolve(t(CK), crossprod(Xi, yi)))
        S_hat <- S_i + as.numeric(crossprod(yi) - t(th) %*% K %*% th)/2

        # add the contribution of equation i
        lml <- lml - 0.5*(sum(log(Vi)) + 2*sum(log(diag(CK)))) +
            lgamma(nu_i + T/2) + nu_i*log(S_i) - lgamma(nu_i) -
            (nu_i + T/2)*log(S_hat)
    }
    lml
}
