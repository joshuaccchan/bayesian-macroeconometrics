# inefficiency_factor.R
# Computes inefficiency factors (integrated autocorrelation times) for
# MCMC output. For each column of draws, IF[j] = Omega_j / sigma2_j,
# where sigma2_j is the marginal variance and Omega_j is the long-run
# variance estimated by the spectral variance at zero (Bartlett window).
# Requires specvar0.R.
#
# Inputs:
#   draws : R-by-k matrix of MCMC draws (each column is a scalar sequence);
#           a plain vector is treated as a single column
#   L     : truncation lag for the spectral variance estimator
#
# Output:
#   IF    : length-k vector of inefficiency factors

inefficiency_factor <- function(draws, L) {
    if (is.null(dim(draws))) draws <- matrix(draws, ncol = 1)
    R <- nrow(draws)
    k <- ncol(draws)
    IF <- numeric(k)
    for (j in 1:k) {
        x <- draws[, j]
        sigma2 <- var(x)*(R-1)/R      # MATLAB var(x,1): population version
        Omega  <- specvar0(x, L) * R  # long-run variance of x^{(r)}
        IF[j] <- Omega / sigma2
    }
    IF
}
