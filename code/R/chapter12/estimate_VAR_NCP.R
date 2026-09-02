# estimate_VAR_NCP.R
# Computes the posterior hyperparameters of a VAR(p) under the natural
# conjugate normal-inverse-Wishart prior, building the regressor matrix from
# the sample Y and the pre-sample observations Y0.
#
# Inputs:
#   Y   : T x n matrix of observations
#   Y0  : p0 x n matrix of pre-sample observations (p0 >= p)
#   p   : lag order
#   A0  : k x n prior mean of the coefficient matrix A, k = 1+n*p
#   VA  : k x k prior scale matrix, so that Cov(vec(A)|Sigma) = Sigma x VA
#   nu0 : prior degrees of freedom for the inverse-Wishart on Sigma
#   S0  : n x n prior scale matrix for the inverse-Wishart on Sigma
#
# Outputs (returned in a list):
#   A_hat  : k x n posterior mean of A
#   KA     : k x k posterior precision matrix
#   nu_hat : posterior degrees of freedom
#   S_hat  : n x n posterior scale matrix

estimate_VAR_NCP <- function(Y, Y0, p, A0, VA, nu0, S0) {
    T <- nrow(Y)
    n <- ncol(Y)
    k <- 1 + n*p

    # construct the T x k regressor matrix Z whose t-th row is
    # x_t' = (1, y_{t-1}', ..., y_{t-p}')
    tmpY <- rbind(Y0[(nrow(Y0)-p+1):nrow(Y0), , drop = FALSE], Y)
    Z <- matrix(0, T, n*p)
    for (i in 1:p) {
        Z[, ((i-1)*n+1):(i*n)] <- tmpY[(p-i+1):(nrow(tmpY)-i), , drop = FALSE]
    }
    Z <- cbind(1, Z)

    # compute posterior hyperparameters
    iVA <- solve(VA)   # VA^{-1}
    KA <- iVA + crossprod(Z)
    # Cholesky factorize KA once and obtain A_hat by two triangular solves;
    # R's chol() returns the UPPER factor, so KA = t(CKA) %*% CKA
    CKA <- chol(KA)
    A_hat <- backsolve(CKA, forwardsolve(t(CKA), iVA %*% A0 + crossprod(Z, Y)))
    S_hat <- S0 + t(A0) %*% iVA %*% A0 + crossprod(Y) - t(A_hat) %*% KA %*% A_hat
    # symmetrize to correct for rounding errors
    S_hat <- (S_hat + t(S_hat))/2
    nu_hat <- nu0 + T

    list(A_hat = A_hat, KA = KA, nu_hat = nu_hat, S_hat = S_hat)
}
