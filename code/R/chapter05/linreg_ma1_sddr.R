# linreg_ma1_sddr.R
# Computes the Bayes factor for the ARMA(2,1) model against the AR(2)
# model with independent errors via the Savage-Dickey density ratio.
# The prior for psi is uniform on (-1, 1), so the prior ordinate at
# psi = 0 is 0.5. The posterior ordinate at psi = 0 is estimated by
# averaging the conditional posterior densities p(psi | y, beta, sig2)
# over the posterior draws of (beta, sig2) produced by linreg_ma1.R.
#
# NOTE: this script evaluates the MA(1) likelihood nsim x n_grid times
# (about 20 million times with the shipped settings) and takes on the
# order of half an hour in R.

source("loglike_MA1.R")

source("linreg_ma1.R")  # estimate the ARMA(2,1) model

n_grid <- 400  # number of grid points
psi_grid <- seq(-.99, .99, length.out = n_grid)  # grid for psi
psi_grid <- sort(c(psi_grid, 0))  # insert 0 explicitly
n_grid <- length(psi_grid)
idx_0 <- which(psi_grid == 0)  # index for psi = 0
lp_psi <- numeric(n_grid)      # log posterior density
store_p_psi <- numeric(n_grid)

for (isim in 1:nsim) {
    beta <- store_theta[isim, 1:3]
    sig2 <- store_theta[isim, 4]

    for (igrid in 1:n_grid) {
        psi <- psi_grid[igrid]
        lp_psi[igrid] <- loglike_MA1(psi,
            y - X %*% beta, sig2)
    }
    # normalize to obtain conditional posterior of psi
    p_psi <- exp(lp_psi - max(lp_psi))
    # trapezoidal rule (MATLAB's trapz(psi_grid, p_psi))
    p_psi <- p_psi/sum(diff(psi_grid)*(p_psi[-1] + p_psi[-n_grid])/2)
    store_p_psi <- store_p_psi + p_psi
}
p_psi_hat <- store_p_psi/nsim   # posterior of psi
BF_UR <- 0.5/p_psi_hat[idx_0]   # BF in favor of MA(1)
cat(sprintf("Bayes factor in favor of MA(1): %.3f\n",
    BF_UR))

# plot prior and posterior densities
prior_psi <- rep(0.5, n_grid)
par(mfrow = c(1, 1), mar = c(4.5, 4.5, 1, 1))
plot(psi_grid, p_psi_hat, type = "l", col = "black", lwd = 2,
     ylim = c(0, max(p_psi_hat)),
     xlab = expression(psi), ylab = "Density")
lines(psi_grid, prior_psi, col = "black", lwd = 1.5, lty = 2)
legend("topright", legend = c("Posterior", "Prior"),
       lty = c(1, 2), lwd = c(2, 1.5), bty = "n")
