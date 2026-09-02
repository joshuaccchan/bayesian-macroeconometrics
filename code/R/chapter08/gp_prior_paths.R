# gp_prior_paths.R
# Visualize Gaussian process priors through sample paths.
#
#   Draws a handful of sample paths from a zero-mean Gaussian process under
#   two covariance functions and plots them side by side in a 1-by-2 panel:
#
#     (left)  Brownian motion kernel      k(x,x') = min(x,x')
#             -> continuous but rough (nowhere differentiable) paths
#     (right) squared exponential kernel  k(x,x') = sf2*exp(-(x-x')^2/(2*l^2))
#             -> smooth paths
#
#   The kernel is the object that encodes prior beliefs about regularity and
#   persistence: the squared exponential produces smooth curves, while the
#   Brownian motion kernel produces jagged ones. This reproduces the
#   "Visualizing a Gaussian Process" illustration in the Gaussian process
#   regression section. Paths are drawn in grayscale so the figure reads in
#   black and white.
#
# The .m version is a zero-argument function acting as a driver; here it is a
# top-to-bottom script with the local helper gp_sample defined first.

gp_sample <- function(K, npaths) {
    # Draw NPATHS sample paths from N(0,K) via the Cholesky factor, adding a
    # small jitter to the diagonal (increased adaptively) for numerical
    # stability. R's chol() errors out where MATLAB's returns the flag p > 0,
    # so the test is a tryCatch; t(chol(.)) is MATLAB's lower factor.
    n      <- nrow(K)
    jitter <- 1e-12
    L      <- NULL
    while (is.null(L)) {
        L <- tryCatch(t(chol(K + jitter*diag(n))), error = function(e) NULL)
        if (is.null(L)) jitter <- jitter*10   # not yet positive definite: add more
    }
    F <- L %*% matrix(rnorm(n*npaths), n, npaths)
    F
}

set.seed(42)                            # fix the seed for reproducible paths

x      <- seq(0, 1, length.out = 200)   # input grid on [0,1]
npaths <- 5                             # number of sample paths per panel

# --- covariance (Gram) matrices on the grid ---
K_bm   <- outer(x, x, pmin)             # Brownian motion kernel
sf2    <- 1                             # signal variance of the SE kernel
lambda <- 0.15                          # length scale of the SE kernel
K_se   <- sf2*exp(-outer(x, x, "-")^2/(2*lambda^2))   # squared exponential kernel

# --- draw sample paths  f ~ N(0,K)  (each column is one path) ---
F_bm <- gp_sample(K_bm, npaths)
F_se <- gp_sample(K_se, npaths)

# --- plot the 1-by-2 panel (black-and-white, grayscale paths) ---
fs   <- 1.15                                    # font scale
lw   <- 1                                       # line width
grays <- gray(seq(0, 0.7, length.out = npaths)) # black -> light gray
yl   <- 1.05*max(abs(c(F_bm, F_se)))            # common vertical scale

setEPS()
postscript("gp_prior_paths.eps", width = 8, height = 3.2)
par(mfrow = c(1, 2), mar = c(4.2, 4.2, 2.4, 1), bty = "l", las = 1,
    cex.axis = fs, cex.lab = fs, cex.main = fs)

plot(NA, xlim = c(0, 1.1), ylim = c(-yl, yl),
     xlab = expression(x), ylab = expression(f(x)),
     main = expression(k[BM]))
for (j in 1:npaths) {
    lines(x, F_bm[, j], lty = 1, col = grays[j], lwd = lw)
}

plot(NA, xlim = c(0, 1.1), ylim = c(-yl, yl),
     xlab = expression(x), ylab = expression(f(x)),
     main = expression(k[SE]))
for (j in 1:npaths) {
    lines(x, F_se[, j], lty = 1, col = grays[j], lwd = lw)
}

# Export for the book
invisible(dev.off())

cat(sprintf("%d sample paths per panel on a grid of %d points; common y-scale = %.3f\n",
            npaths, length(x), yl))
cat("figure written to gp_prior_paths.eps\n")
