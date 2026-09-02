# fit_BayesRidge.R
# Returns the posterior-mean ridge estimator under an isotropic
# Gaussian prior beta_j ~ N(0, 1/lambda) and known unit error
# variance. The estimator is beta_hat = (lambda*I + X'X)^{-1} X'y.
#
# Inputs:
#   y     : length-T response
#   X     : T-by-k design matrix
#   lambda: ridge regularisation parameter (prior precision scale)
#
# Output:
#   beta_hat: length-k ridge posterior mean

fit_BayesRidge <- function(y, X, lambda) {
    k <- ncol(X)
    XX <- crossprod(X)
    Xy <- crossprod(X, y)
    Dbeta <- solve(lambda*diag(k) + XX)
    beta_hat <- as.numeric(Dbeta %*% Xy)
    beta_hat
}
