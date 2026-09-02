# loglike_MA1.R
# Gaussian log-likelihood of the linear regression model with MA(1)
# errors, evaluated on regression residuals e = y - X*beta. The MA(1)
# structure is enforced via the band matrix H_psi, and the whitened
# residuals u = H_psi^{-1} e are obtained by a banded back-solve.
#
# Inputs:
#   psi : scalar MA(1) parameter
#   e   : length-T vector of regression residuals (y - X*beta)
#   sig2: innovation variance
#
# Outputs (returned as a list):
#   loglik: log p(e | psi, sig2) evaluated using u = H_psi^{-1} e
#   u     : implied innovations u
#
# Requires the Matrix package (bandSparse replaces MATLAB's spdiags;
# Hpsi is unit lower bidiagonal, so solve() does a banded back-solve).

loglike_MA1 <- function(psi, e, sig2) {
    T <- length(e)
    Hpsi <- bandSparse(T, T, k = c(0, -1),
                       diagonals = list(rep(1, T), rep(psi, T - 1)))
    u <- as.numeric(solve(Hpsi, e))
    loglik <- -0.5*T*log(2*pi*sig2) - 0.5*sum(u^2)/sig2
    list(loglik = loglik, u = u)
}
