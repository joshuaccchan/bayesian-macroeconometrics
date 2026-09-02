# griddy_gibbs.R
# Generic Griddy-Gibbs sampler for a univariate target density with
# bounded support (a,b), given an unnormalized log-density logf(x).
#
# Inputs:
#   logf  : function returning log unnormalized density at x;
#           must accept a vector input and return a vector output of
#           the same size
#   a, b  : lower and upper bounds (a < b), finite
#   n_grid: number of grid points (integer >= 2)
#
# Outputs (returned as a list):
#   x_draw   : one draw from the Griddy-Gibbs approximation to the target
#   x_grid   : grid points used
#   logf_grid: logf evaluated on x_grid

griddy_gibbs <- function(logf, a, b, n_grid) {
    # jittered grid to avoid repeated draws
    step <- (b - a) / (n_grid + 1)   # grid spacing
    x_grid <- a + step*(1:n_grid)
    x_grid <- x_grid + (runif(n_grid) - 0.5)*step   # small jitter
        # enforce strict bounds
    eps <- .Machine$double.eps
    x_grid <- pmin(pmax(x_grid, a + eps), b - eps)

    # evaluate logf on the grid and normalize
    logf_grid <- logf(x_grid)
    w <- exp(logf_grid - max(logf_grid))
    w <- w / sum(w)   # normalize to sum to 1

    cdf_grid <- cumsum(w)
    u <- runif(1)
    idx <- which(cdf_grid >= u)[1]
    x_draw <- x_grid[idx]

    list(x_draw = x_draw, x_grid = x_grid, logf_grid = logf_grid)
}
