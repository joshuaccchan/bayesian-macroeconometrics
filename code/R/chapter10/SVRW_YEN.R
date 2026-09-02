# SVRW_YEN.R
# Collapsed Gibbs sampler for the standard (random-walk) stochastic volatility
# model, fitted to daily YEN/USD returns. The model is
#   y_t = exp(h_t/2)*eps_t,   eps_t ~ N(0, 1),
#   h_t = h_{t-1} + u_t,      u_t   ~ N(0, sigh2),
# with priors h0 ~ N(a0, b0) and sigh2 ~ IG(nu_h, S_h).
#
# Requires: SVRW.R, SV_RW_gaussian_approx.R

suppressMessages(library(Matrix))
source("SVRW.R")
source("SV_RW_gaussian_approx.R")

set.seed(42)   # for reproducibility
nsim <- 20000; burnin <- 1000
data <- read.csv("YENUSD.csv", header = FALSE)[, 1]
y <- data; T <- length(y)

# prior hyperparameters
a0 <- 0; b0 <- 100
nu_h <- 3; S_h <- .2^2*(nu_h-1)

# storage
store_theta <- matrix(0, nsim, 2)   # [h0 sigh2]
store_h <- matrix(0, nsim, T)

# precompute a few things
S1 <- sparseMatrix(i = 2:T, j = 1:(T-1), x = 1, dims = c(T, T))
H <- Diagonal(T) - S1
HH <- crossprod(H)
c <- 1e-4; ystar <- log(y^2 + c)

# initialize
sigh2 <- .05
h0 <- log(var(y))
h <- SV_RW_gaussian_approx(y^2, h0, sigh2)
for (isim in 1:(nsim + burnin)) {
    # sample sigh2
    # R's rgamma takes shape and scale
    sigh2 <- 1/rgamma(1, shape = nu_h + T/2,
                      scale = 1/(S_h + as.numeric(crossprod(h-h0, HH %*% (h-h0)))/2))

    # sample h0
    Kh0 <- 1/b0 + 1/sigh2
    h0_hat <- (a0/b0 + h[1]/sigh2)/Kh0
    h0 <- h0_hat + 1/sqrt(Kh0)*rnorm(1)

    # sample h
    h <- SVRW(ystar, h, h0, sigh2)

    if (isim > burnin) {
        isave <- isim - burnin
        store_h[isave, ] <- h
        store_theta[isave, ] <- c(h0, sigh2)
    }
}
h_std <- exp(store_h/2)   # transform to standard deviation
h_mean <- colMeans(h_std)
h_CI <- apply(h_std, 2, quantile, probs = c(0.05, 0.95), type = 5)
h_lower <- h_CI[1, ]
h_upper <- h_CI[2, ]

theta_mean <- colMeans(store_theta)
theta_CI <- apply(store_theta, 2, quantile, probs = c(.05, .95), type = 5)

cat("posterior means of [h0, sigh2]:", theta_mean, "\n")
cat("90% credible intervals (rows: 5%, 95%):\n")
print(theta_CI)

tt <- seq(2005, 2013, length.out = T)
setEPS()
postscript("YENdata.eps", width = 9, height = 3.5)
par(mar = c(4, 4, 1, 1), cex.axis = 1.2, cex.lab = 1.2)
plot(tt, y, type = "l", col = "black", lwd = 1.5, xlim = c(2005, 2013),
     ylim = c(-6, 6), yaxt = "n", bty = "n", xlab = "Time", ylab = "")
axis(2, at = seq(-6, 6, by = 2))
dev.off()

setEPS()
postscript("SV_YEN_h.eps", width = 9, height = 3.5)
par(mar = c(4, 4, 1, 1), cex.axis = 1.2, cex.lab = 1.2)
plot(tt, h_mean, type = "n", xlim = c(2005, 2013),
     ylim = range(c(h_lower, h_upper)), bty = "n", xlab = "Time", ylab = "")
polygon(c(tt, rev(tt)), c(h_lower, rev(h_upper)), col = gray(0.8), border = NA)
lines(tt, h_mean, col = "black", lwd = 1.5)
dev.off()
