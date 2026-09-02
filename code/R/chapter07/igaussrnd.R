# igaussrnd.R
# Draws from the inverse Gaussian distribution IGAUSS(psi, mu) with
# kernel
#   f(x) \propto x^{-3/2} exp{ -psi (x-mu)^2 / (2 x mu^2) },  x > 0.
# Uses the standard chi-squared-based representation from
# Michael, Schucany, and Haas (1976).
#
# Inputs:
#   psi : positive shape parameter (scalar or length-n vector)
#   mu  : positive mean parameter  (scalar or length-n vector)
#   n   : number of draws (defaults to max(length(psi), length(mu)))
#
# Output:
#   x   : length-n vector of draws from IGAUSS(psi, mu)

igaussrnd <- function(psi, mu, n = NULL) {
    if (is.null(n)) {
        n <- max(length(psi), length(mu))
    }

    # expand scalars to length n; force plain vectors
    if (length(psi) == 1) psi <- rep(psi, n) else psi <- as.vector(psi)
    if (length(mu)  == 1) mu  <- rep(mu,  n) else mu  <- as.vector(mu)

    if (length(psi) != n || length(mu) != n) {
        stop("psi and mu must be scalars or vectors of the same length n.")
    }

    # step 1: nu0 ~ chi^2_1
    nu0 <- rnorm(n)^2

    # step 2: candidate draws
    sqrt_term <- sqrt(4*mu*psi*nu0 + (mu^2)*(nu0^2))
    x1 <- mu + (mu^2)*nu0/(2*psi) - (mu/(2*psi))*sqrt_term
    x2 <- (mu^2)/x1

    # step 3: accept/reject switch
    p <- mu/(mu + x1)
    U <- runif(n) < p
    x <- ifelse(U, x1, x2)
    x
}
