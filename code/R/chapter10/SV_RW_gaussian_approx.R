# SV_RW_gaussian_approx.R
# Posterior mean of the log-volatility vector h under a single-Gaussian
# (moment-matching) approximation to log(chi^2_1) and a random-walk prior for
# h_t. The transformed observation y*_t = log(y_t^2 + c) is treated as
# N(h_t - 1.2704, 4.9348), giving a linear Gaussian state space model whose
# posterior mean is obtained by solving a banded linear system.
#
# Inputs:
#   s2    : length-T vector of squared observations y_t^2
#   h0    : initial log-volatility (state at time 0), scalar
#   sigh2 : innovation variance of the random-walk state equation, scalar
#
# Output:
#   h_hat : length-T posterior mean of the log-volatility

suppressMessages(library(Matrix))

SV_RW_gaussian_approx <- function(s2, h0, sigh2) {

    c <- 1e-4        # avoids log(0)
    mu_eps <- -1.2704   # E[log chi^2_1]
    var_eps <- 4.9348   # Var[log chi^2_1]
    T <- length(s2)
    S1 <- sparseMatrix(i = 2:T, j = 1:(T-1), x = 1, dims = c(T, T))
    H <- Diagonal(T) - S1

    ystar <- log(s2 + c)
    P <- crossprod(H)/sigh2   # prior precision
    b <- rep(h0, T)           # prior mean
    Kh <- P + Diagonal(T)/var_eps
    h_hat <- as.numeric(solve(Kh, as.numeric(P %*% b) + (ystar - mu_eps)/var_eps))
    h_hat
}
