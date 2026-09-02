# UCSV.R
# Gibbs sampler for the unobserved components model with stochastic volatility
# of Stock and Watson (2007), fitted to quarterly PCE inflation. The
# model is
#   y_t   = tau_t + eps^y_t,         eps^y_t   ~ N(0, exp(h_t)),
#   tau_t = tau_{t-1} + eps^tau_t,   eps^tau_t ~ N(0, exp(g_t)),
# with random-walk log-volatilities h_t (gap) and g_t (trend), each with an
# inverse-gamma prior on its innovation variance.
#
# Requires: SVRW.R, SV_RW_gaussian_approx.R

suppressMessages(library(Matrix))
source("SVRW.R")
source("SV_RW_gaussian_approx.R")

set.seed(42)   # for reproducibility
nsim <- 50000; burnin <- 1000

# load PCE data - 1960Q1-2024Q4 (column PCECTPI, 260 rows)
data <- read.csv("USPCE.csv")$PCECTPI
y <- data
T <- length(y)

# prior hyperparameters
a0_h <- 0; b0_h <- 10          # h0 ~ N(a0_h, b0_h)
a0_g <- 0; b0_g <- 10          # g0 ~ N(a0_g, b0_g)
a0_tau <- 0; b0_tau <- 10      # tau0 ~ N(a0_tau, b0_tau)
nu_oh <- 3; S_oh <- 0.2^2*(nu_oh-1)   # omega_h^2 ~ IG(nu_oh, S_oh)
nu_og <- 3; S_og <- 0.2^2*(nu_og-1)   # omega_g^2 ~ IG(nu_og, S_og)

# precompute a few things
c <- 1e-4   # log-squared safeguard
S1 <- sparseMatrix(i = 2:T, j = 1:(T-1), x = 1, dims = c(T, T))
H  <- Diagonal(T) - S1
HH <- crossprod(H)

# initialize
tau0 <- mean(y)
tau  <- tau0*rep(1, T)
h0 <- log(var(y)); g0 <- log(var(y))
omega_h2 <- 0.1;  omega_g2 <- 0.1
    # initialize h using Gaussian approximation
h <- SV_RW_gaussian_approx((y - tau)^2, h0, omega_h2)
    # initialize g using Gaussian approximation
dtau <- tau - c(tau0, tau[1:(T-1)])
g <- SV_RW_gaussian_approx(dtau^2, g0, omega_g2)

# storage
    #[omega_h2 omega_g2 h0 g0 tau0]
store_theta <- matrix(0, nsim, 5)
store_tau <- matrix(0, nsim, T)
store_h <- matrix(0, nsim, T)
store_g <- matrix(0, nsim, T)

for (isim in 1:(nsim+burnin)) {

    # sample tau
    iOh <- Diagonal(x = exp(-h))   # Omega_h^{-1}
    HiOgH <- t(H) %*% Diagonal(x = exp(-g)) %*% H
    Ktau <- forceSymmetric(HiOgH + iOh)
    tau_mean <- solve(Ktau, as.numeric(HiOgH %*% (tau0*rep(1, T))) +
                            as.numeric(iOh %*% y))
    tau <- as.numeric(tau_mean + solve(chol(Ktau), rnorm(T)))

    # sample tau0
    Ktau0 <- 1/b0_tau + exp(-g[1])
    tau0_hat <- (a0_tau/b0_tau + tau[1]*exp(-g[1]))/Ktau0
    tau0 <- tau0_hat + (1/sqrt(Ktau0))*rnorm(1)

    # sample h
    ystar_h <- log((y - tau)^2 + c)
    h <- SVRW(ystar_h, h, h0, omega_h2)

    # sample omega_h2
    # R's rgamma takes shape and scale
    omega_h2 <- 1/rgamma(1, shape = nu_oh + T/2,
        scale = 1/(S_oh + 0.5*as.numeric(crossprod(h - h0, HH %*% (h - h0)))))

    # sample h0
    Kh0 <- 1/b0_h + 1/omega_h2
    h0_hat <- (a0_h/b0_h + h[1]/omega_h2)/Kh0
    h0 <- h0_hat + (1/sqrt(Kh0))*rnorm(1)

    # sample g
    dtau <- tau - c(tau0, tau[1:(T-1)])
    ystar_g <- log(dtau^2 + c)
    g <- SVRW(ystar_g, g, g0, omega_g2)

    # sample omega_g2
    omega_g2 <- 1/rgamma(1, shape = nu_og + T/2,
        scale = 1/(S_og + 0.5*as.numeric(crossprod(g - g0, HH %*% (g - g0)))))

    # sample g0
    Kg0 <- 1/b0_g + 1/omega_g2
    g0_hat <- (a0_g/b0_g + g[1]/omega_g2)/Kg0
    g0 <- g0_hat + (1/sqrt(Kg0))*rnorm(1)

    if (isim > burnin) {
        isave <- isim - burnin
        store_tau[isave, ] <- tau
        store_h[isave, ] <- h
        store_g[isave, ] <- g
        store_theta[isave, ] <- c(omega_h2, omega_g2, h0, g0, tau0)
    }
}
# posterior summaries
theta_mean <- colMeans(store_theta)
tau_mean <- colMeans(store_tau)
h_mean <- colMeans(exp(store_h/2))
g_mean <- colMeans(exp(store_g/2))

cat("posterior means of [omega_h2, omega_g2, h0, g0, tau0]:\n")
print(theta_mean)

tt <- seq(1960, 2024.75, by = .25)
setEPS()
postscript("UCSV_trend.eps", width = 9, height = 3.5)
par(mar = c(4, 4, 1, 1), cex.axis = 1.2, cex.lab = 1.2)
plot(tt, y, type = "l", lty = 2, lwd = 1.2, col = "black",   # data
     xlim = c(1959.75, 2025), ylim = range(c(y, tau_mean)),
     bty = "n", xlab = "Time", ylab = "")
lines(tt, tau_mean, lty = 1, lwd = 1.8, col = "black")       # trend
legend("topright", legend = c("Actual inflation", "Trend inflation"),
       lty = c(2, 1), lwd = c(1.2, 1.8), col = "black", bty = "n", cex = 1.1)
dev.off()

setEPS()
postscript("UCSV_SV.eps", width = 9, height = 3.5)
par(mar = c(4, 4, 1, 1), cex.axis = 1.2, cex.lab = 1.2)
plot(tt, h_mean, type = "l", lty = 1, lwd = 1.8, col = "black",   # gap volatility
     xlim = c(1959.75, 2025), ylim = range(c(h_mean, g_mean)),
     bty = "n", xlab = "Time", ylab = "")
lines(tt, g_mean, lty = 2, lwd = 1.5, col = "black")              # trend volatility
legend("topright", legend = c("Gap volatility", "Trend volatility"),
       lty = c(1, 2), lwd = c(1.8, 1.5), col = "black", bty = "n", cex = 1.1)
dev.off()
