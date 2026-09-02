# UC_AR.R
# Gibbs sampler for a local level (unobserved components) model of US CPI
# inflation with an AR(1) cyclical component. The model is
#   y_t   = tau_t + eps_t,
#   eps_t = rho*eps_{t-1} + u_t,    u_t   ~ N(0, sig2),
#   tau_t = tau_{t-1} + eta_t,      eta_t ~ N(0, omega2),
# with priors tau0 ~ N(a0, b0), rho ~ U(-1, 1), and inverse-gamma priors on
# sig2 and omega2. The trend tau is drawn in one block from its Gaussian full
# conditional using a precision (band-matrix) sampler; rho is drawn from a
# truncated normal via tnormrnd.R; and sig2, omega2, and tau0 from standard
# conjugate updates.
#
# Requires: tnormrnd.R

suppressMessages(library(Matrix))
source("tnormrnd.R")

set.seed(42)
nsim <- 50000; burnin <- 1000
# load data - US CPI 1948M1 - 2019M12 (column CPIAUCSL, first 864 rows)
data <- read.csv("USCPI.csv")$CPIAUCSL[1:864]
y <- data
T <- length(y)

# initialize for storage
store_tau <- matrix(0, nsim, T)
store_theta <- matrix(0, nsim, 4)   # [rho,sig2,omega2,tau0]

# prior hyperparameters
a0 <- 5; b0 <- 100
nu_sig0 <- 3; S_sig0 <- 1*(nu_sig0-1)
nu_omega0 <- 3; S_omega0 <- .25^2*(nu_omega0-1)

# initialize the Markov chain
sig2 <- 1; omega2 <- .1; tau0 <- 5; rho <- 0

# compute a few things outside the loop
S1 <- sparseMatrix(i = 2:T, j = 1:(T-1), x = 1, dims = c(T,T)) # first-lag shift matrix
H <- Diagonal(T) - S1
H_rho <- Diagonal(T) - rho*S1
HH_rho <- crossprod(H_rho)
HH <- crossprod(H)
HHiota <- as.numeric(HH %*% rep(1,T))

for (isim in 1:(nsim+burnin)) {
    # sample tau
    Ktau <- HH/omega2 + HH_rho/sig2
    tau_hat <- solve(Ktau, tau0/omega2*HHiota + as.numeric(HH_rho %*% y)/sig2)
    # Matrix's chol() returns the UPPER factor with no fill-reducing
    # permutation, so it is MATLAB's chol(Ktau,'lower')' directly
    Ctau <- chol(Ktau)
    tau <- as.numeric(tau_hat + solve(Ctau, rnorm(T)))

    # sample rho
    e <- y - tau
    Krho <- sum(e[1:(T-1)]^2)/sig2
    rho_hat <- sum(e[1:(T-1)]*e[2:T])/sum(e[1:(T-1)]^2)
    rho <- tnormrnd(rho_hat, 1/Krho, -1, 1)
    H_rho <- Diagonal(T) - rho*S1
    HH_rho <- crossprod(H_rho)

    # sample sig2
    u <- as.numeric(H_rho %*% e)
    # R's rgamma takes shape and scale
    sig2 <- 1/rgamma(1, shape = nu_sig0 + T/2, scale = 1/(S_sig0 + sum(u^2)/2))

    # sample omega2
    d <- tau - tau0
    omega2 <- 1/rgamma(1, shape = nu_omega0 + T/2,
        scale = 1/(S_omega0 + as.numeric(crossprod(d, HH %*% d))/2))

    # sample tau0
    Ktau0 <- 1/b0 + 1/omega2
    tau0_hat <- (a0/b0 + tau[1]/omega2)/Ktau0
    tau0 <- tau0_hat + sqrt(1/Ktau0)*rnorm(1)

    if (isim > burnin) {
        isave <- isim - burnin
        store_tau[isave,] <- tau
        store_theta[isave,] <- c(rho, sig2, omega2, tau0)
    }
}
tau_mean <- colMeans(store_tau)
theta_mean <- colMeans(store_theta)
tau_q  <- apply(store_tau, 2, quantile, probs = c(0.05, 0.95), type = 5)
tau_lo <- tau_q[1,]
tau_hi <- tau_q[2,]

cat("posterior means of [rho, sig2, omega2, tau0]:\n")
print(theta_mean)

# monthly time axis: 1948M1 to 2019M12
tt <- seq(as.Date("1948-01-01"), by = "1 month", length.out = T)
tt_lo <- seq(tt[1], by = "-12 months", length.out = 2)[2]
tt_hi <- seq(tt[T], by = "12 months", length.out = 2)[2]

setEPS()
postscript("CPI_trend.eps", width = 9, height = 3.5)
par(mar = c(4.5, 4.5, 1, 1))
plot(tt, tau_mean, type = "n", xlim = c(tt_lo, tt_hi),
     ylim = c(min(c(y, tau_lo)) - 1, max(c(y, tau_hi)) + 1),
     xlab = "Time", ylab = "Inflation", cex.lab = 1.4, cex.axis = 1.4,
     bty = "l")

polygon(c(tt, rev(tt)), c(tau_lo, rev(tau_hi)), col = gray(0.85), border = NA)

lines(tt, tau_mean, col = "black", lwd = 1.5)
lines(tt, y, col = "black", lty = 3, lwd = 1)
box(bty = "l")

legend("topright", legend = c("Trend", "Inflation"),
       lty = c(1, 3), lwd = c(1.5, 1), col = "black", bty = "n", cex = 1.4)
invisible(dev.off())
