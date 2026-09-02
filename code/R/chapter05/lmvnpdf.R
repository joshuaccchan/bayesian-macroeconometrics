# lmvnpdf.R
# Evaluates the log density of a multivariate normal
# distribution
#
# Inputs:
# x: evaluation points, k x 1
# mu: mean vector, k x 1
# Sig: covariance matrix, k x k
#
# Output:
# logden: log density of N(mu,Sig) evaluated at x

lmvnpdf <- function(x, mu, Sig) {
    x <- as.numeric(x)
    mu <- as.numeric(mu)
    k <- length(mu)

    # R's chol() returns the UPPER factor, so transpose it to obtain
    # the lower Cholesky factor of MATLAB's chol(Sig,'lower')
    CSig <- t(chol(as.matrix(Sig)))
    e <- forwardsolve(CSig, x - mu)

    logden <- -0.5*k*log(2*pi) - sum(log(diag(CSig))) - 0.5*sum(e^2)
    logden
}
