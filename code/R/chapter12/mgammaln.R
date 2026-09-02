# mgammaln.R
# Evaluates the log of the multivariate gamma function, log Gamma_n(x).
#
# Inputs:
#   n : dimension
#   x : argument
#
# Output:
#   k : log of the multivariate gamma function evaluated at x

mgammaln <- function(n, x) {
    n*(n-1)/4*log(pi) + sum(lgamma(x + seq(0, (1-n)/2, by = -0.5)))
}
