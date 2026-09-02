# gpr_uncertainty.R
# Empirical-Bayes Gaussian process regression of quarterly US real GDP growth
# on its own lag and lagged macroeconomic uncertainty. The model is
#   y_t = f(x_t) + eps_t,  eps_t ~ N(0, sig2),  f ~ GP(0, k),
#   x_t = (y_{t-1}, U_{t-1})',
# with an ARD squared exponential kernel
#   k(x,x') = sigf2 * exp(-0.5 * sum_j (x_j - x'_j)^2 / lam_j^2).
# The hyperparameters (sigf2, lam_1, lam_2, sig2) are set by maximizing the
# log integrated likelihood (gpr_fit_eb.R); the fitted mean surface is sliced
# at low/median/high lagged growth and plotted against uncertainty. The
# uncertainty index is from Jurado-Ludvigson-Ng (2015); sample 1960Q4-2019Q4.
#
# Requires: gpr_fit_eb.R, gpr_predict.R, shaded_band.R

source("gpr_fit_eb.R")
source("gpr_predict.R")
source("shaded_band.R")

set.seed(42)

# load data; keep the pre-COVID sample (through 2019Q4)
tbl <- read.csv("gpr_uncertainty_data.csv", stringsAsFactors = FALSE)
yr <- as.numeric(sub("Q.*$", "", tbl$date))
qt <- as.numeric(sub("^.*Q", "", tbl$date))
tbl <- tbl[(yr + (qt-1)/4) <= 2019.9, ]

y <- tbl$gdp_growth
X <- cbind(tbl$gdp_growth_lag1, tbl$macro_unc_h1_lag1)  # [lagged growth, lagged unc]
n <- nrow(X)

# center response (zero-mean GP) and standardize each input
my <- mean(y);  yc <- y - my
mx <- colMeans(X);  sx <- apply(X, 2, sd)
Xs <- sweep(sweep(X, 2, mx, "-"), 2, sx, "/")

# empirical-Bayes hyperparameters (maximize the log integrated likelihood)
opts <- list(nstart = 12, verbose = FALSE)
fit <- gpr_fit_eb(yc, Xs, opts)
hp  <- fit$hp
llf <- fit$logintlike

nm <- c("lagged GDP growth", "lagged uncertainty")
cat(sprintf("Empirical-Bayes ARD hyperparameters (n = %d, %s-%s)\n",
            n, tbl$date[1], tbl$date[n]))
cat(sprintf("  signal variance sig_f^2 = %.4f\n", hp$sigf2))
for (j in 1:2) {
    cat(sprintf("  length scale lambda_%d = %6.3f (std)  = %7.4f (%s units)\n",
                j, hp$lam[j], hp$lam[j]*sx[j], nm[j]))
}
cat(sprintf("  noise variance sig^2 = %.4f (sig = %.3f)\n", hp$sig2, sqrt(hp$sig2)))
cat(sprintf("  log integrated likelihood = %.3f\n", llf))
less <- which.max(hp$lam)
cat(sprintf("  -> %s has the larger length scale (less relevant locally)\n", nm[less]))

# slices: fix lagged growth, vary uncertainty
# (type = 5 is the Hazen rule, which matches MATLAB's quantile)
gvals <- quantile(X[, 1], c(0.10, 0.50, 0.90), type = 5)  # low / median / high growth
glab  <- c("10th pct", "median", "90th pct")
ug    <- seq(min(X[, 2]), max(X[, 2]), length.out = 300)  # uncertainty grid

z95 <- 1.96
sty <- c(3, 1, 2)                            # low / median / high (: - --)

# 95% credible band for the median slice
Xq  <- sweep(sweep(cbind(gvals[2]*rep(1, length(ug)), ug), 2, mx, "-"), 2, sx, "/")
pm  <- gpr_predict(yc, Xs, Xq, hp)
mum <- pm$mu
sdm <- sqrt(pm$s2f)
band_lo <- mum + my - z95*sdm
band_hi <- mum + my + z95*sdm

# fitted slices (base R needs the axes before the shading, so the three
# curves are computed first and drawn after the band)
mu_s <- matrix(0, length(ug), 3)
for (s in 1:3) {
    Xq <- sweep(sweep(cbind(gvals[s]*rep(1, length(ug)), ug), 2, mx, "-"), 2, sx, "/")
    mu_s[, s] <- gpr_predict(yc, Xs, Xq, hp)$mu + my
}

setEPS()
postscript("gpr_uncertainty.eps", width = 9.17, height = 3.61)
par(mar = c(4.2, 4.4, 1, 1), bty = "l", las = 1, cex.axis = 1.15, cex.lab = 1.15)
plot(NA, xlim = c(min(X[, 2]), max(X[, 2])),
     ylim = range(c(band_lo, band_hi, mu_s)), xaxs = "i",
     xlab = "Macro uncertainty", ylab = "Real GDP growth")
shaded_band(ug, band_lo, band_hi, 0.85)
for (s in 1:3) {
    lines(ug, mu_s[, s], col = "black", lty = sty[s], lwd = 2)
}
leg <- list(
    bquote(y[t-1] == .(sprintf("%.1f%%", gvals[1])) ~ "(10%-tile)"),
    bquote(y[t-1] == .(sprintf("%.1f%%", gvals[2])) ~ "(50%-tile)"),
    bquote(y[t-1] == .(sprintf("%.1f%%", gvals[3])) ~ "(90%-tile)"))
legend("bottomleft", legend = as.expression(leg), lty = sty, lwd = 2,
       col = "black", bty = "n", cex = 1.05)

# export figure
invisible(dev.off())
