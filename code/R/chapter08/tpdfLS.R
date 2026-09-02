# tpdfLS.R
# Univariate Student-t density in location-scale form, evaluated at y
# with location mu, scale s2, and nu degrees of freedom. The variance is
# nu/(nu-2)*s2 for nu > 2.

tpdfLS <- function(y, mu, s2, nu) {
    s <- sqrt(s2)
    z <- (y - mu)/s
    # the gamma ratio is computed via lgamma: a direct gamma(.)/gamma(.)
    # overflows to Inf/Inf = NaN once nu exceeds about 343, which happens
    # in the DPM of Chapter 8 as soon as a cluster holds ~340 observations
    c <- exp(lgamma((nu+1)/2) - lgamma(nu/2))/(sqrt(nu*pi)*s)
    f <- c*(1 + (z^2)/nu)^(-(nu+1)/2)
    f
}
