# logintlike_SFM.R
# Evaluates the log integrated likelihood of the static factor model, i.e., the
# Gaussian density of the data with the latent factors integrated out:
#   y_t ~ N(0, A*Omega*A' + Sigma).
# The n x n inverse is obtained via the Sherman-Morrison-Woodbury identity, and
# the log-determinant and quadratic form are computed from the Cholesky factor
# of the precision matrix, avoiding direct inversion of an n x n matrix.
#
# Inputs:
#   Y     : data, T x n
#   a     : free elements of the lower-triangular loading matrix A
#   Sig   : idiosyncratic variances, n x 1
#   Omega : factor variances, r x 1
#
# Output:
#   loglike : log integrated likelihood

logintlike_SFM <- function(Y, a, Sig, Omega) {
    T <- nrow(Y)
    n <- ncol(Y)
    r <- length(Omega)

    # construct unit lower-triangular A
    A <- rbind(diag(r), matrix(0, n - r, r))
    count_a <- 0
    for (ii in 2:n) {
        nai <- min(ii - 1, r)
        A[ii, 1:nai] <- a[(count_a + 1):(count_a + nai)]
        count_a <- count_a + nai
    }

    # diagonal precision matrices; n and r are small, so dense diagonals are
    # used here where the MATLAB uses spdiags
    iSig <- diag(1/as.numeric(Sig), nrow = n)
    iOmega <- diag(1/as.numeric(Omega), nrow = r)

    # compute (A*Omega*A' + Sig)^{-1} using Woodbury identity
    AiSig <- t(A) %*% iSig
    iB <- iSig - t(AiSig) %*% solve(iOmega + AiSig %*% A, AiSig)

    CiB <- t(chol(iB))  # Cholesky factor of precision (lower)
    CY <- t(CiB) %*% t(Y)
    quad <- sum(CY^2)  # quadratic term: sum_t y_t' iB y_t
    -0.5*T*n*log(2*pi) + T*sum(log(diag(CiB))) - 0.5*quad
}
