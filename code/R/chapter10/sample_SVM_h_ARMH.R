# sample_SVM_h_ARMH.R
# One MCMC update of the log-volatility vector h in the stochastic volatility
# in mean model, using a Laplace-based acceptance-rejection Metropolis-Hastings
# step. A Gaussian proposal N(h_hat, Kh^{-1}) is built from a second-order
# Taylor expansion of the log conditional posterior about its mode (located by
# Newton-Raphson); candidates are drawn by acceptance-rejection screening and
# then corrected by an independence-chain MH step.
#
# Inputs:
#   y     : length-T vector of observations
#   alp   : volatility-in-mean coefficient (scalar or length-T vector)
#   mu    : conditional-mean component x_t'beta (scalar or length-T vector)
#   h     : length-T current draw of the log-volatility
#   h0    : initial log-volatility (state at time 0), scalar
#   sigh2 : innovation variance of the random-walk state equation, scalar
#   HH    : T-by-T sparse prior precision component H'*H of the random-walk prior
#
# Outputs:
#   a list with elements
#   h      : length-T updated draw of the log-volatility
#   accept : 1 if the final MH step accepts the proposal, 0 otherwise

suppressMessages(library(Matrix))

sample_SVM_h_ARMH <- function(y, alp, mu, h, h0, sigh2, HH) {

    T <- length(h)
    accept <- 0
    kappa <- 3

    # step 1: locate posterior mode by Newton-Raphson
    s2 <- (y - mu)^2
    ht <- h   # initial value for Newton-Raphson
    max_norm <- Inf
    tol <- 1e-4
    max_iter <- 100; iter <- 0   # safeguard: log-posterior is concave, so
                                 # Newton-Raphson converges in a few steps
    Kh <- NULL
    while (max_norm > tol && iter < max_iter) {
        iter <- iter + 1
        exp_ht <- exp(ht)
        curv1 <- 0.5*alp^2*exp_ht
        curv2 <- 0.5*s2/exp_ht

        g <- -0.5 - curv1 + curv2
        grad <- g - as.numeric(HH %*% (ht - h0))/sigh2   # gradient

        G  <- Diagonal(x = -curv1 - curv2)
        Kh <- HH/sigh2 - G   # negative Hessian

        new_ht <- ht + as.numeric(solve(Kh, grad))
        max_norm <- max(abs(new_ht - ht))
        ht <- new_ht
    }
    h_hat <- ht

    # step 2: construct Gaussian approximation
    # N(h_hat, Kh^{-1})
    Ch <- chol(Kh)   # upper factor: Kh = t(Ch) %*% Ch
    logdetKh <- 2*sum(log(diag(Ch)))

    # log posterior kernel (normalizing constant omitted)
    logp <- function(x) as.numeric(
        -0.5*crossprod(x - h0, HH %*% (x - h0))/sigh2 - 0.5*sum(x) -
         0.5*sum(exp(-x)*(y - mu - alp*exp(x))^2))

    # Log Gaussian proposal density g(h)
    logg <- function(x) as.numeric(
        -0.5*T*log(2*pi) + 0.5*logdetKh -
         0.5*crossprod(x - h_hat, Kh %*% (x - h_hat)))

    # step 3: choose c = kappa * p(h_hat)/g(h_hat)
    logc <- log(kappa) + logp(h_hat) - logg(h_hat)

    # proposal kernel: q(h) \propto  min{p(h)/(c g(h)), 1} g(h)
    logq <- function(x) min(logp(x) - logc - logg(x), 0) + logg(x)

    # step 4: acceptance-rejection screening from g
    accepted_AR <- FALSE
    while (!accepted_AR) {
        hc <- as.numeric(h_hat + solve(Ch, rnorm(T)))   # draw from g
        log_acc_AR <- logp(hc) - logc - logg(hc)
        if (log(runif(1)) < min(log_acc_AR, 0))
            accepted_AR <- TRUE
    }

    # step 5: MH correction
    log_alpha_ARMH <- logp(hc) - logp(h) + logq(h) - logq(hc)
    if (log(runif(1)) < log_alpha_ARMH) {
        h <- hc
        accept <- 1
    }
    list(h = h, accept = accept)
}
