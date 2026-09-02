# sample_CSV_h_ARMH.R
# Samples the common log-volatility h = (h_1,...,h_T)' in a VAR with a common
# stochastic volatility error structure (Section 14.2) using a Laplace-based
# acceptance-rejection Metropolis-Hastings (ARMH) step. The log-volatility
# follows the stationary AR(1) h_t = phi*h_{t-1} + u_t^h, u_t^h ~ N(0,sigh2),
# with h_1 ~ N(0, sigh2/(1-phi^2)). Given the per-period sums of squares s2
# (summed over the n variables), the conditional density of h is non-standard.
# A Gaussian approximation centered at the posterior mode serves as the
# proposal: a candidate is first screened by acceptance-rejection and then
# corrected by a Metropolis-Hastings step.
#
# Inputs:
#   s2:    length-T vector of per-period sums of squares
#   phi:   AR(1) persistence of the log-volatility
#   sigh2: innovation variance of the log-volatility
#   h:     length-T current draw of the log-volatility
#   n:     number of variables in the VAR
#   kappa: (optional) envelope constant for the AR screening step; larger
#          values improve mixing at the cost of more candidate draws (default 3)
#
# Outputs (returned as a list):
#   h:      length-T new draw of the log-volatility
#   accept: 1 if the MH step accepts, 0 otherwise

suppressMessages(library(Matrix))

sample_CSV_h_ARMH <- function(s2, phi, sigh2, h, n, kappa = 3) {
    T <- length(h)
    accept <- 0

    # AR(1) prior precision HiSH = Hphi' * diag(.) * Hphi (zero mean)
    Hphi <- Diagonal(T) - phi*sparseMatrix(i = 2:T, j = 1:(T-1), x = 1,
                                           dims = c(T, T))
    HiSH <- forceSymmetric(crossprod(Hphi,
        Diagonal(x = c((1-phi^2)/sigh2, rep(1/sigh2, T-1))) %*% Hphi))

    # step 1: locate the posterior mode by Newton-Raphson
    ht <- h
    max_norm <- Inf
    tol <- 1e-4
    while (max_norm > tol) {
        eis2 <- exp(-ht)*s2
        grad <- as.numeric(-(HiSH %*% ht) - n/2 + 0.5*eis2)  # gradient of log posterior
        Kh <- HiSH + Diagonal(x = 0.5*eis2)                  # negative Hessian
        new_ht <- ht + as.numeric(solve(Kh, grad))
        max_norm <- max(abs(new_ht - ht))
        ht <- new_ht
    }
    h_hat <- ht

    # step 2: construct the Gaussian approximation N(h_hat, Kh^{-1})
    Ch <- chol(Kh)                    # UPPER factor: Kh = t(Ch) %*% Ch
    logdetKh <- 2*sum(log(diag(Ch)))

    # log posterior kernel (normalizing constant omitted)
    logp <- function(x) as.numeric(-0.5*crossprod(x, HiSH %*% x) - n/2*sum(x) -
                                   0.5*sum(exp(-x)*s2))

    # log Gaussian proposal density g(h)
    logg <- function(x) as.numeric(-0.5*T*log(2*pi) + 0.5*logdetKh -
                                   0.5*crossprod(x - h_hat, Kh %*% (x - h_hat)))

    # step 3: choose c = kappa * p(h_hat)/g(h_hat)
    logc <- log(kappa) + logp(h_hat) - logg(h_hat)

    # proposal kernel: q(h) \propto min{p(h)/(c g(h)), 1} g(h)
    logq <- function(x) min(logp(x) - logc - logg(x), 0) + logg(x)

    # step 4: acceptance-rejection screening from g
    accepted_AR <- FALSE
    while (!accepted_AR) {
        hc <- h_hat + as.numeric(solve(Ch, rnorm(T)))   # draw from g
        if (log(runif(1)) < min(logp(hc) - logc - logg(hc), 0)) {
            accepted_AR <- TRUE
        }
    }

    # step 5: Metropolis-Hastings correction
    log_alpha <- logp(hc) - logp(h) + logq(h) - logq(hc)
    if (log(runif(1)) < log_alpha) {
        h <- hc
        accept <- 1
    }

    list(h = h, accept = accept)
}
