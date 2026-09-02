# mcse.R
# Computes Monte Carlo standard errors (MCSEs) for posterior means
# obtained from MCMC output. For each column of draws, MCSE[j] is
# sqrt(Omega_j / R), where Omega_j is the long-run variance estimated
# by the spectral variance at zero. Requires specvar0.R.
#
# Inputs:
#   draws : R-by-k matrix of MCMC draws
#   L     : truncation lag for the spectral variance estimator
#
# Output:
#   MCSE  : k-vector of Monte Carlo standard errors

source("specvar0.R")

mcse <- function(draws, L) {
    # a plain vector is treated as a single column (MATLAB size gives R-by-1)
    if (is.null(dim(draws))) draws <- matrix(draws, ncol = 1)
    R <- nrow(draws)
    k <- ncol(draws)
    MCSE <- numeric(k)
    for (j in 1:k) {
        x <- draws[, j]
        Omega <- specvar0(x, L) * R   # long-run variance of x^{(r)}
        MCSE[j] <- sqrt(Omega/R)
    }
    MCSE
}
