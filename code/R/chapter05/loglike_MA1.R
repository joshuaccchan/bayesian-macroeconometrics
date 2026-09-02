# loglike_MA1.R
# Gaussian log-likelihood of the linear regression model with MA(1)
# errors, evaluated on regression residuals e = y - X*beta. The MA(1)
# structure is enforced via the band matrix H_psi, and the whitened
# residuals u = H_psi^{-1} e are obtained by a banded back-solve.
#
# Inputs:
#   psi : scalar MA(1) parameter
#   e   : T-vector of regression residuals (y - X*beta)
#   sig2: innovation variance
#
# Output:
#   loglik: log p(e | psi, sig2) evaluated using u = H_psi^{-1} e
#           (the .m version also returns the innovations u as a second
#           output, but no script in this chapter uses it)

loglike_MA1 <- function(psi, e, sig2) {
    e <- as.numeric(e)
    T <- length(e)
    # H_psi is lower bidiagonal with 1 on the main diagonal and psi on
    # the first subdiagonal, so u = H_psi \ e is the forward recursion
    #   u_1 = e_1,   u_t = e_t - psi*u_{t-1},  t = 2,...,T.
    # The .m builds H_psi with spdiags and calls the sparse solver; in R
    # the same solve is done with stats::filter, which is exact and much
    # cheaper because this function is called millions of times in
    # linreg_ma1_sddr.R (building a sparse T-by-T matrix per call would
    # dominate the runtime). For the equivalent explicit band matrix see
    # H_psi in linreg_ma1.R.
    u <- as.numeric(stats::filter(e, filter = -psi, method = "recursive"))
    loglik <- -0.5*T*log(2*pi*sig2) - 0.5*sum(u^2)/sig2
    loglik
}
