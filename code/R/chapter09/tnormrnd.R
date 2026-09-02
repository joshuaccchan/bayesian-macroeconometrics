# This function draws from a univariate truncated normal distribution using
# the inverse-transform method. It returns N draws from a normal
# distribution with mean mu and variance sigma2, truncated to the interval (a,b)
#
# Inputs:
# mu: mean (scalar or length-N vector)
# sigma2: variance (scalar or length-N vector)
# a: lower truncation point (scalar)
# b: upper truncation point (scalar)
# N: number of draws (optional; defaults to length(mu))
#
# Output:
# draws: length-N vector of truncated normal draws
#        (a length-1 vector behaves as a scalar, as MATLAB's 1-by-1 does)

tnormrnd <- function(mu, sigma2, a, b, N = NULL) {

    # check number of inputs
    if (missing(mu) || missing(sigma2) || missing(a) || missing(b))
        stop("Wrong number of arguments.")

    # length of mean vector
    K <- length(mu)

    # if N not supplied, set equal to length(mu)
    if (is.null(N))
        N <- K

    # dimension check
    if ((K != N || length(sigma2) != N) && K != 1)
        stop("Dimensions of mu and sigma2 must equal N.")

    # expand scalars to vectors if necessary
    if (K == 1) {
        mu     <- rep(mu, N)
        sigma2 <- rep(sigma2, N)
    }

    sigma <- sqrt(sigma2)
    u <- runif(N)

    # compute CDF values at truncation points
    p1 <- pnorm((a - mu)/sigma)
    p2 <- pnorm((b - mu)/sigma)

    # apply inverse CDF transformation
    C <- qnorm(p1 + (p2 - p1)*u)

    # transform back to truncated normal draw
    draws <- mu + sigma*C
    draws
}
