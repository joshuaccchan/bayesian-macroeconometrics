# sample_psi_RW.R
# Random-walk Metropolis-Hastings update for the MA(1) parameter psi
# in a linear regression model with MA(1) errors. The invertibility
# restriction |psi| < 1 is enforced by assigning zero acceptance
# probability to proposals outside the admissible region.
#
# Inputs:
#   psi  : current value of the MA(1) parameter
#   e    : length-T vector of regression residuals (y - X*beta)
#   sig2 : innovation variance sigma^2
#   g_var: proposal variance for the random-walk step
#
# Outputs (returned as a list):
#   psi   : updated value of the MA(1) parameter
#   accept: acceptance indicator (TRUE if accepted, FALSE otherwise)
#
# Requires loglike_MA1.R.

sample_psi_RW <- function(psi, e, sig2, g_var) {
    psi_c <- psi + sqrt(g_var)*rnorm(1)   # propose candidate

        # check invertibility (0.999 is a numerical buffer)
    if (abs(psi_c) < 0.999) {
        log_alpha <- loglike_MA1(psi_c, e, sig2)$loglik -
            loglike_MA1(psi,   e, sig2)$loglik
    } else {
        log_alpha <- -Inf
    }

    accept <- (log(runif(1)) < log_alpha)   # accept/reject step
    if (accept) {
        psi <- psi_c
    }
    list(psi = psi, accept = accept)
}
