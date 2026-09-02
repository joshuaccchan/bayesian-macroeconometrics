# ml_VAR_NCP.R
# Evaluates the log marginal likelihood of a VAR under the natural conjugate
# (normal-inverse-Wishart) prior, computed in logs for numerical stability.
#
# Requires: mgammaln.R, ldet.R
#
# Inputs:
#   VA    : k x k prior scale matrix, so that Cov(vec(A)|Sigma) = Sigma x VA
#   S0    : n x n prior scale matrix for the inverse-Wishart on Sigma
#   nu0   : prior degrees of freedom for the inverse-Wishart
#   KA    : k x k posterior precision matrix, KA = VA^{-1} + Z'Z
#   S_hat : n x n posterior scale matrix
#   T     : number of observations
#
# Output:
#   lml : log marginal likelihood

ml_VAR_NCP <- function(VA, S0, nu0, KA, S_hat, T) {
    n <- nrow(S0)
    lml <- -n*T/2*log(pi) - n/2*(ldet(VA) + ldet(KA)) + nu0/2*ldet(S0) -
        (nu0+T)/2*ldet(S_hat) + mgammaln(n, (nu0+T)/2) - mgammaln(n, nu0/2)
    lml
}
