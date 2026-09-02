# UC_output_gap.R
# Gibbs sampler for a local linear trend (unobserved components) model that
# decomposes 100*log US real GDP into trend (potential) output and a cyclical
# output gap. The model is
#   y_t       = tau_t + c_t,
#   c_t       = phi_1*c_{t-1} + phi_2*c_{t-2} + u_t^c,   u_t^c   ~ N(0, sigc2),
#   D.tau_t   = D.tau_{t-1} + u_t^tau,                   u_t^tau ~ N(0, sigtau2),
# where D denotes the first difference, so trend growth follows a random walk.
# The trend tau is drawn with a precision (band-matrix) sampler; phi by an
# accept-reject step enforcing stationarity; sigc2 and (tau0, tau_{-1})
# from conjugate updates; and sigtau2 with a Griddy-Gibbs step via griddy_gibbs.R.
#
# Requires: griddy_gibbs.R, shade_nber_recessions.R

suppressMessages(library(Matrix))
source("griddy_gibbs.R")
source("shade_nber_recessions.R")

set.seed(42)
nsim <- 20000; burnin <- 1000
# load data - US GDP 1947Q1 - 2019Q4 (column GDPC1, first 292 rows)
data_raw <- read.csv("USGDP.csv")$GDPC1[1:292]
data <- 100*log(data_raw)
y <- data
T <- length(y)

# prior hyperparameters
a0 <- c(750, 750); B0 <- 100*diag(2)
phi0 <- c(1.3, -.7); iVphi <- diag(2)
nu_sigc2 <- 3; S_sigc2 <- 1*(nu_sigc2-1)
sigtau2_ub <- .01

# storage
store_theta <- matrix(0, nsim, 6)   # [phi, sigc2, sigtau2, tau0]
store_tau <- matrix(0, nsim, T)
store_mu <- matrix(0, nsim, T)      # annualized trend growth

# initialize
phi <- c(1.34, -.7)
tau0 <- c(y[1], y[1])   # [tau_{0}, tau_{-1}]
sigc2 <- .5
sigtau2 <- .001

# construct a few things
S1 <- sparseMatrix(i = 2:T, j = 1:(T-1), x = 1, dims = c(T,T)) # first-lag shift matrix
S2 <- sparseMatrix(i = 3:T, j = 1:(T-2), x = 1, dims = c(T,T)) # second-lag shift matrix
H2 <- Diagonal(T) - 2*S1 + S2
H2H2 <- crossprod(H2)
Hphi <- Diagonal(T) - phi[1]*S1 - phi[2]*S2
Xtau0 <- cbind(2:(T+1), -(1:T))
n_grid <- 500
count_phi <- 0

for (isim in 1:(nsim+burnin)) {
    # sample tau
    r <- numeric(T)
    r[1] <- 2*tau0[1] - tau0[2]
    r[2] <- -tau0[1]
    alp_tau <- as.numeric(solve(H2, r))
    HpHp <- crossprod(Hphi)
    Ktau <- H2H2/sigtau2 + HpHp/sigc2
    tau_hat <- solve(Ktau, as.numeric(H2H2 %*% alp_tau)/sigtau2 +
                           as.numeric(HpHp %*% y)/sigc2)
    # Matrix's chol() returns the UPPER factor with no fill-reducing
    # permutation, so it is MATLAB's chol(Ktau,'lower')' directly
    tau <- as.numeric(tau_hat + solve(chol(Ktau), rnorm(T)))

    # sample phi
    cyc <- y - tau   # the cycle; called c in the .m, renamed since c() is
                     # R's concatenate function
    Xphi <- cbind(c(0, cyc[1:(T-1)]), c(0, 0, cyc[1:(T-2)]))
    Kphi <- iVphi + crossprod(Xphi)/sigc2
    phi_hat <- solve(Kphi, as.numeric(iVphi %*% phi0) +
                           as.numeric(crossprod(Xphi, cyc))/sigc2)
    phic <- as.numeric(phi_hat + solve(chol(Kphi), rnorm(2)))
    if (sum(phic) < .99 && phic[2] - phic[1] < .99 &&
            phic[2] > -.99) {
        phi <- phic
        Hphi <- Diagonal(T) - phi[1]*S1 - phi[2]*S2
        count_phi <- count_phi + 1
    }

    # sample sigc2
    ec <- cyc - as.numeric(Xphi %*% phi)
    # R's rgamma takes shape and scale
    sigc2 <- 1/rgamma(1, shape = nu_sigc2 + T/2,
        scale = 1/(S_sigc2 + sum(ec^2)/2))

    # sample sigtau2 via Griddy-Gibbs on (0, sigtau2_ub)
    del_tau <- c(tau0[1], tau) -
        c(tau0[2], tau0[1], tau[1:(T-1)])
        # first differences of tau
    ddel_tau <- del_tau[2:length(del_tau)] - del_tau[1:(length(del_tau)-1)]
        # second differences of tau
    logf_sigtau2 <- function(x) -(T/2)*log(x) -
        sum(ddel_tau^2)/(2*x)
    sigtau2 <- griddy_gibbs(logf_sigtau2, 1e-12,
        sigtau2_ub, n_grid)$x_draw

    # sample tau0
    Ktau0 <- solve(B0) + crossprod(Xtau0, as.matrix(H2H2 %*% Xtau0))/sigtau2
    tau0_hat <- solve(Ktau0, solve(B0, a0) +
        as.numeric(crossprod(Xtau0, as.numeric(H2H2 %*% tau)))/sigtau2)
    tau0 <- as.numeric(tau0_hat + solve(chol(Ktau0), rnorm(2)))

    if (isim > burnin) {
        i <- isim - burnin
        store_tau[i,] <- tau
        store_theta[i,] <- c(phi, sigc2, sigtau2, tau0)
        store_mu[i,] <- 4*(tau - c(tau0[1], tau[1:(T-1)]))
    }
}
tau_mean <- colMeans(store_tau)
theta_mean <- colMeans(store_theta)
theta_CI <- apply(store_theta, 2, quantile, probs = c(.025, .975), type = 5)
mu_mean <- colMeans(store_mu)

cat("posterior means of [phi1, phi2, sigc2, sigtau2, tau0, tau_{-1}]:\n")
print(theta_mean)
cat("95% credible intervals (rows: 2.5%, 97.5%):\n")
print(theta_CI)

# plot of graphs
tt <- seq(1947, 2019.75, by = .25)

y_gap <- y - tau_mean
yl <- c(min(y_gap) - 1, max(y_gap) + 1)

setEPS()
postscript("UC_gap.eps", width = 9, height = 3.5)
par(mar = c(4.5, 4.5, 1, 1))
plot(tt, y_gap, type = "n", xlim = c(1947, 2020), ylim = yl,
     xlab = "Time", ylab = "Output gap", cex.lab = 1.4, cex.axis = 1.4,
     bty = "l")
shade_nber_recessions(yl[1], yl[2])

lines(tt, y_gap, col = "black", lwd = 1.5)
lines(tt, numeric(T), col = "black", lty = 2, lwd = 1)
box(bty = "l")
invisible(dev.off())


yl <- c(1, 4.5)
setEPS()
postscript("UC_trend_growth.eps", width = 9, height = 3.5)
par(mar = c(4.5, 4.5, 1, 1))
plot(tt, mu_mean, type = "n", xlim = c(1947, 2020), ylim = yl,
     xlab = "Time", ylab = "Output trend growth", cex.lab = 1.4, cex.axis = 1.4,
     bty = "l")
shade_nber_recessions(yl[1], yl[2])
lines(tt, mu_mean, col = "black", lwd = 1.5)
box(bty = "l")
invisible(dev.off())
