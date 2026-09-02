# specvar0.R
# Estimates the long-run variance of the sample mean of x using a
# Bartlett-window spectral-variance estimator at frequency zero:
#       S = (gamma_0 + 2 * sum_{ell=1}^{L} w_ell * gamma_ell) / T,
# where gamma_ell is the lag-ell sample autocovariance and the weights
# w_ell = 1 - ell/(L+1) are Bartlett (Newey-West) weights.
#
# Inputs:
#   x : T-vector (demeaned internally)
#   L : truncation lag (nonnegative integer)
#
# Output:
#   S : long-run variance of the sample mean of x

specvar0 <- function(x, L) {
    x <- as.numeric(x)
    T <- length(x)
    x <- x - mean(x)

    # autocovariances up to lag L
    gamma0 <- sum(x * x)/T
    Sx <- gamma0

    # seq_len() so that L = 0 gives an empty loop (R's 1:0 is c(1, 0)); lags
    # ell >= T are dropped because MATLAB's x(1+ell:end) is empty there, so
    # they contribute nothing, whereas R's x[(1+ell):T] would run backwards
    for (ell in seq_len(min(L, T - 1))) {
        w <- 1 - ell/(L + 1)   # Bartlett weight
        gamma <- sum(x[(1 + ell):T] * x[1:(T - ell)])/T
        Sx <- Sx + 2 * w * gamma
    }

    # long-run variance of the mean
    S <- Sx/T
    S
}
