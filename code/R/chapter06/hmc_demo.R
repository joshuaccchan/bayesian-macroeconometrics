# hmc_demo.R
# Demonstrates Hamiltonian Monte Carlo sampling from the bivariate
# target f(theta_1, theta_2) ~
#   exp(-(theta_2 - theta_1^2)^2/20 - (theta_1 - 1)^2/2).
# Generates nsim post burn-in draws using L leapfrog steps with step
# size eps and standard Gaussian momentum.
# Requires leapfrog.R.

source("leapfrog.R")

set.seed(42)

# log target density and its gradient
logf <- function(q) -0.05*(q[2] - q[1]^2)^2 -
    0.5*(q[1] - 1)^2
grad_logf <- function(q) c(q[1]/5*(q[2] - q[1]^2) -
    (q[1] - 1), -(q[2] - q[1]^2)/10)

# HMC settings
nsim <- 5000; burnin <- 1000
eps <- 0.8; L <- 20

# storage and initialization
theta <- rnorm(2)
samples <- matrix(0, nsim, 2)
accepts <- 0

# main HMC loop
for (isim in 1:(nsim + burnin)) {
    p0 <- rnorm(2)

    # propose using L-step leapfrog
    prop <- leapfrog(theta, p0, eps, L, grad_logf)
    thetac <- prop$thetaNew; pc <- prop$pNew

    # Hamiltonians at current and proposed states
    H0 <- -logf(theta)  + 0.5*sum(p0^2)
    Hc <- -logf(thetac) + 0.5*sum(pc^2)

    # MH accept/reject step. A divergent leapfrog trajectory can overflow
    # to Inf/NaN; isTRUE() makes such a proposal simply fail the test
    # (as it does in MATLAB) instead of raising an error in R's if().
    if (isTRUE(log(runif(1)) < -(Hc - H0))) {
        theta <- thetac
        if (isim > burnin) {
            accepts <- accepts + 1
        }
    }
    if (isim > burnin) {
        samples[isim - burnin, ] <- theta
    }
}

cat(sprintf("HMC acceptance rate: %.3f\n", accepts/nsim))

# plot: samples and target density contours
t1 <- samples[, 1]
t2 <- samples[, 2]

xg <- seq(min(t1), max(t1), length.out = 300)
yg <- seq(min(t2), max(t2), length.out = 300)

# Z[i,j] = density at (xg[i], yg[j]) -- R's contour() takes the grid
# this way round rather than via meshgrid
Z <- outer(xg, yg, function(T1, T2)
    exp(-(0.05*(T2 - T1^2)^2 + 0.5*(T1 - 1)^2)))
Z <- Z/max(Z)

contour(xg, yg, Z, nlevels = 12, drawlabels = FALSE, col = "black",
        frame.plot = FALSE,
        xlab = expression(theta[1]), ylab = expression(theta[2]),
        cex.lab = 1.4)
points(t1, t2, pch = 20, cex = 0.3, col = "black")
