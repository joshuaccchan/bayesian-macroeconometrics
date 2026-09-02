# VAR_ACP_kappa.R
# Selects the own-lag (kappa2) and other-lag (kappa3) shrinkage
# hyperparameters of the asymmetric conjugate prior for the four-variable
# VAR(7) on macro4_Q by maximizing the closed-form marginal likelihood over
# a grid, and plots the normalized marginal-likelihood contours over the
# two hyperparameters.
#
# Requires: ml_VAR_ACP.R

source("ml_VAR_ACP.R")

set.seed(42)   # for reproducibility

# load data
data <- as.matrix(read.csv("macro4_Q.csv")[, -1])   # drop date column -> n = 4
p <- 7
Y0 <- data[1:8, ]      # pre-sample (initial conditions)
Y <- data[9:nrow(data), ]   # estimation sample 1962Q1-2019Q4
T <- nrow(Y); n <- ncol(Y)
kappa1 <- 100          # intercept: weak shrinkage

# residual variances from univariate AR(p) models
s2 <- numeric(n); tmpY <- rbind(Y0[(nrow(Y0)-p+1):nrow(Y0), ], Y)
for (i in 1:n) {
    Xi <- matrix(1, T, 1)
    for (l in 1:p) Xi <- cbind(Xi, tmpY[(p-l+1):(nrow(tmpY)-l), i])
    e <- Y[, i] - Xi %*% qr.solve(Xi, Y[, i])
    s2[i] <- sum(e^2)/(T - ncol(Xi))
}

# evaluate the log marginal likelihood over a grid of (kappa2, kappa3)
k2g <- seq(0.005, 0.60, length.out = 90)      # own-lag shrinkage
k3g <- seq(0.0005, 0.10, length.out = 90)     # other-lag shrinkage
LML <- matrix(0, length(k3g), length(k2g))
for (a in seq_along(k2g)) {
    for (b in seq_along(k3g)) {
        LML[b, a] <- ml_VAR_ACP(Y, Y0, p, c(kappa1, k2g[a], k3g[b]), s2)
    }
}
dens <- exp(LML - max(LML))   # normalized surface (flat prior), max = 1

# locate the maximizer and the best symmetric (kappa2 = kappa3) value
ia <- which.max(apply(LML, 2, max)); ib <- which.max(LML[, ia])
fsym <- function(lk) -ml_VAR_ACP(Y, Y0, p, c(kappa1, exp(lk), exp(lk)), s2)
ks <- exp(optimize(fsym, c(log(1e-4), log(1)))$minimum)
cat(sprintf("asymmetric optimum (own,other) = (%.3f, %.4f)\n", k2g[ia], k3g[ib]))
cat(sprintf("best symmetric = %.3f;   log-ML gain over best symmetric = %.2f\n",
            ks, max(LML) + fsym(log(ks))))

# plot the marginal-likelihood contours with the key points marked
setEPS()
postscript("VAR_ACP_kappa.eps", width = 5.6, height = 4.2)
par(mar = c(4.5, 4.5, 1, 1))
# R's contour() wants z with dim (length(x), length(y)), hence the transpose
contour(k2g, k3g, t(dens), nlevels = 12, col = "black", drawlabels = FALSE,
        xlim = c(0, 0.6), ylim = c(0, 0.1), bty = "n",
        xlab = expression(kappa[2] ~ "(own lags)"),
        ylab = expression(kappa[3] ~ "(other lags)"),
        cex.lab = 1.3, cex.axis = 1.1)
lines(c(0, 0.1), c(0, 0.1), lty = 2, col = gray(0.5), lwd = 1.3)  # symmetric
points(0.04, 0.04, pch = 21, bg = "white", col = "black", cex = 1.6,
       lwd = 1.2)   # natural conjugate, 0.2^2
points(k2g[ia], k3g[ib], pch = 8, col = "black", cex = 1.8, lwd = 2)  # maximizer
invisible(dev.off())
