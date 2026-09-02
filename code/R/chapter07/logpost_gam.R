# logpost_gam.R
# Evaluates the log of the collapsed posterior kernel p(gam | y, sig2)
# for the SSVS regression, with the regression coefficients beta
# integrated out analytically. The returned value sums the marginal
# log-likelihood log p(y | gam, sig2) and the log-prior log p(gam)
# under independent Bernoulli(bp) inclusion priors. If the implied
# precision matrix is not positive definite, returns -Inf.
#
# Inputs:
#   gam   : length-k vector of inclusion indicators in {0,1}
#   y     : length-T response
#   X     : T-by-k design matrix
#   sig2  : scalar error variance
#   bp    : length-k vector of prior inclusion probabilities
#   iVbeta: k-by-k prior precision of beta
#
# Output:
#   logden: log p(gam | y, sig2) up to a normalising constant

logpost_gam <- function(gam, y, X, sig2, bp, iVbeta) {
    k <- length(gam)
    Xtilde <- sweep(X, 2, gam, "*")   # X %*% diag(gam): zero out excluded columns

    iDbeta <- iVbeta + crossprod(Xtilde)/sig2
    # R's chol() is upper triangular, like MATLAB's chol; it throws an
    # error instead of returning MATLAB's flag when iDbeta is not p.d.
    C <- tryCatch(chol(iDbeta), error = function(e) NULL)
    if (is.null(C)) {
        return(-Inf)
    }

    beta_hat <- as.numeric(solve(iDbeta, crossprod(Xtilde, y)/sig2))
    logden <- -sum(log(diag(C))) + 0.5*sum(beta_hat*(iDbeta %*% beta_hat)) +
        sum(gam[2:k]*log(bp[2:k]) + (1-gam[2:k])*log(1-bp[2:k]))
    logden
}
