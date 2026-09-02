# geweke_diag.R
# Computes Geweke's convergence diagnostic for MCMC output. For each
# column of draws, the chain is split into an early segment A (first
# fraction a) and a late segment B (last fraction 1-b), the segment
# means are compared, and the Z-statistic
#       Z = (mean_A - mean_B) / sqrt(S_A + S_B)
# is computed, where S_A and S_B are spectral-variance estimates of the
# long-run variances of the two segment means. Under stationarity, Z is
# approximately standard normal; large |Z| provides evidence against
# convergence for the chosen summary. Requires specvar0.R.
#
# Inputs:
#   draws : R-by-k matrix of posterior draws (each column is a scalar
#           summary computed from MCMC output)
#   a     : fraction for early segment (default 0.10)
#   b     : fraction defining late segment, B = floor(b*R):R
#           (default 0.50)
#   Lrule : truncation-lag rule for the spectral variance estimator,
#           either 'auto' (default) or a nonnegative integer L
#
# Outputs (returned in a list):
#   Z     : k-vector of Geweke Z-statistics
#   pval  : k-vector of two-sided p-values (normal approximation)
#   info  : list with elements A, B, LA, LB, a, b giving the segment
#           indices and chosen truncation lags

source("specvar0.R")

geweke_diag <- function(draws, a = 0.10, b = 0.50, Lrule = "auto") {
    # NULL stands in for MATLAB's [] placeholder arguments
    if (is.null(a)) a <- 0.10
    if (is.null(b)) b <- 0.50
    if (is.null(Lrule)) Lrule <- "auto"

    # a plain vector is treated as a single column (MATLAB size gives R-by-1)
    if (is.null(dim(draws))) draws <- matrix(draws, ncol = 1)
    R <- nrow(draws)
    k <- ncol(draws)
    if (!(0 < a && a < b && b < 1)) {
        stop("Require 0 < a < b < 1.")
    }

    A <- seq_len(floor(a * R))
    B <- floor(b * R):R

    nA <- length(A)
    nB <- length(B)

    # Choose truncation lags
    if (is.character(Lrule)) {
        if (tolower(Lrule) == "auto") {
            LA <- max(0, floor(4 * (nA/100)^(2/9)))
            LB <- max(0, floor(4 * (nB/100)^(2/9)))
        } else {
            stop("Unknown Lrule. Use 'auto' or an integer.")
        }
    } else {
        LA <- max(0, floor(Lrule))
        LB <- LA
    }

    Z <- rep(NA_real_, k)
    pval <- rep(NA_real_, k)

    for (j in 1:k) {
        xA <- draws[A, j]
        xB <- draws[B, j]

        mA <- mean(xA)
        mB <- mean(xB)

        SA <- specvar0(xA, LA)   # long-run variance of mean for segment A
        SB <- specvar0(xB, LB)   # long-run variance of mean for segment B

        Z[j] <- (mA - mB)/sqrt(SA + SB)

        # Two-sided p-value under N(0,1)
        pval[j] <- 2 * (1 - pnorm(abs(Z[j])))
    }

    info <- list(A = A, B = B, LA = LA, LB = LB, a = a, b = b)
    list(Z = Z, pval = pval, info = info)
}
