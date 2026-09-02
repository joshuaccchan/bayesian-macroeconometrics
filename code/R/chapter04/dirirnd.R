# dirirnd.R
# Draws from a Dirichlet distribution D(alpha) using the standard
# gamma-normalization construction: if g_m ~ G(alpha_m, 1) iid, then
# (g_1, ..., g_M) / sum_m g_m has a D(alpha) distribution.
#
# Inputs:
#   alpha: length-M vector of concentration parameters (alpha_m > 0)
#   N    : number of draws
#
# Output:
#   W    : N-by-M matrix; each row is an independent draw on the
#          unit simplex

dirirnd <- function(alpha, N) {
    alpha <- as.numeric(alpha)   # ensure alpha is a plain vector
    M <- length(alpha)
    # R's rgamma takes shape and scale; matrix() fills column-major, so
    # rep(alpha, each = N) gives column m the shape alpha[m]
    G <- matrix(rgamma(N*M, shape = rep(alpha, each = N), scale = 1), N, M)
    W <- G / rowSums(G)
    W
}
