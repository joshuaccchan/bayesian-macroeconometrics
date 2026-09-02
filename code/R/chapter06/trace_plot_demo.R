# trace_plot_demo.R
# Simulates three artificial Markov chains -- a well-mixed AR(1), a
# highly autocorrelated AR(1), and a sticky chain that stays put with
# high probability -- and displays their trace plots side by side.

R <- 2200
burnin <- 200
set.seed(0)

# -------------------------------------------------------------------------
# (1) Well-mixed AR(1): low persistence
# -------------------------------------------------------------------------
phi1 <- 0.20;  sig1 <- sqrt(1 - phi1^2)
x1 <- numeric(R)
for (r in 2:R) {
    x1[r] <- phi1*x1[r-1] + sig1*rnorm(1)
}

# -------------------------------------------------------------------------
# (2) Highly autocorrelated AR(1): high persistence
# -------------------------------------------------------------------------
phi2 <- 0.98;  sig2 <- sqrt(1 - phi2^2)
x2 <- numeric(R)
for (r in 2:R) {
    x2[r] <- phi2*x2[r-1] + sig2*rnorm(1)
}

# -------------------------------------------------------------------------
# (3) Sticky chain: stays put with high probability (plateaus)
# -------------------------------------------------------------------------
p_stay <- 0.97
sig3 <- 0.50
x3 <- numeric(R)
for (r in 2:R) {
    if (runif(1) < p_stay) {
        x3[r] <- x3[r-1]
    } else {
        x3[r] <- x3[r-1] + sig3*rnorm(1)
    }
}

# Drop burn-in for display
idx <- (burnin+1):R
x1 <- x1[idx]; x2 <- x2[idx]; x3 <- x3[idx]

# -------------------------------------------------------------------------
# 1x3 panel of trace plots (black and white)
# -------------------------------------------------------------------------
par(mfrow = c(1, 3))

plot(x1, type = "l", col = "black", lwd = 1, bty = "l",
     xlab = "Iteration", ylab = expression(x^(r)), cex.lab = 1.4)
# title('Well-mixed chain')

plot(x2, type = "l", col = "black", lwd = 1, bty = "l",
     xlab = "Iteration", ylab = expression(x^(r)), cex.lab = 1.4)
# title('High autocorrelation')

plot(x3, type = "l", col = "black", lwd = 1, bty = "l",
     xlab = "Iteration", ylab = expression(x^(r)), cex.lab = 1.4)
# title('Sticky chain (stuck)')

# short numerical summary of the three chains (the .m leaves x1, x2, x3
# in the workspace; R scripts have no workspace to inspect afterwards)
lag1 <- function(x) cor(x[-1], x[-length(x)])
cat(sprintf("lag-1 autocorrelation: well-mixed %.3f, persistent %.3f, sticky %.3f\n",
            lag1(x1), lag1(x2), lag1(x3)))
