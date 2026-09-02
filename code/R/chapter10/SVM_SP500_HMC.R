# SVM_SP500_HMC.R
# Gibbs sampler with an HMC update of the log-volatility for the stochastic
# volatility in mean (SVM) model, fitted to daily S&P 500 excess returns. The
# model is
#   y_t = mu + alp*exp(h_t) + eps_t,   eps_t ~ N(0, exp(h_t)),
#   h_t = h_{t-1} + u_t,               u_t   ~ N(0, sigh2),
# with priors gam = (mu, alp)' ~ N(gam0, iVgam^{-1}), sigh2 ~ IG(nu_h, S_h),
# and h0 ~ N(a0, b0). The regression coefficients, sigh2, and h0 use conjugate
# updates; the log-volatility h is sampled by Hamiltonian Monte Carlo using the
# functions logpost_svm_h, grad_logpost_svm_h, and leapfrog below (local
# functions at the end of the MATLAB script; R needs them defined before the
# loop that calls them).
#
# SP500.csv is not distributed with this repository (proprietary); run
# get_SP500_data.R once to create it.
#
# Requires: SV_RW_gaussian_approx.R

suppressMessages(library(Matrix))
source("SV_RW_gaussian_approx.R")

set.seed(3)   # for reproducibility
nsim <- 20000; burnin <- 1000

# prepare the data
sp500_raw <- as.matrix(read.csv("SP500.csv", header = FALSE))   # [date, index level]
dff_raw   <- as.matrix(read.csv("DFF.csv", header = FALSE))     # [date, fed funds rate]
    # keep only valid S&P 500 obs.
valid_idx <- sp500_raw[, 2] != 0
sp500_raw <- sp500_raw[valid_idx, , drop = FALSE]
n_sp <- nrow(sp500_raw)
sp500_dates   <- sp500_raw[2:n_sp, 1]
sp500_returns <- 100*log(sp500_raw[2:n_sp, 2]/sp500_raw[1:(n_sp-1), 2])
    # match return date with the corresponding DFF obs.
dff_loc <- match(sp500_dates, dff_raw[, 1])   # NA where unmatched
is_matched <- !is.na(dff_loc)
sp500_returns <- sp500_returns[is_matched]
dff_loc       <- dff_loc[is_matched]
    # annualized fed funds rate (percent) to daily
y <- sp500_returns - dff_raw[dff_loc, 2]/252
T <- length(y)

# prior hyperparameters
gam0 <- numeric(2); iVgam <- diag(2)/100   # 2-by-2, so kept dense
nu_h <- 3; S_h <- .2^2*(nu_h-1)
a0 <- 0; b0 <- 100

# HMC settings for h
eps <- 0.04   # step size
L <- 20       # # of leapfrog steps
accept_h <- 0 # acceptance counter

# precompute banded prior precision pieces
S1 <- sparseMatrix(i = 2:T, j = 1:(T-1), x = 1, dims = c(T, T))
H  <- Diagonal(T) - S1
HH <- crossprod(H)

# initialize
alp  <- 0
mu <- mean(y)
h0   <- log(var(y))
sigh2 <- .2
    # initialize h using Gaussian approximation
h <- SV_RW_gaussian_approx((y - mu)^2, h0, sigh2)

logpost_svm_h <- function(y, mu, alp, h, h0, sigh2, HH) {
# Log conditional posterior density of the log-volatility h (up to an additive
# constant) in the SVM model, combining the Gaussian likelihood with the
# random-walk prior.
    r <- y - mu - alp*exp(h)
    lden <- -0.5*sum(h) - 0.5*sum((r^2)*exp(-h)) -
        0.5/sigh2*as.numeric(crossprod(h - h0, HH %*% (h - h0)))
    lden
}

grad_logpost_svm_h <- function(y, mu, alp, h, h0, sigh2, HH) {
# Gradient of logpost_svm_h with respect to h (same inputs).
    grad_like <- -0.5 - 0.5*alp^2*exp(h) + 0.5*(y - mu)^2*exp(-h)
    grad_prior <- -as.numeric(HH %*% (h - h0))/sigh2
    grad <- grad_like + grad_prior
    grad
}

leapfrog <- function(h, p, eps, L, grad_logpost) {
# Leapfrog integrator for HMC: L steps of size eps evolving (h, p) along the
# Hamiltonian trajectory, with adjacent half-steps combined (L+1 gradient
# evaluations). grad_logpost is a function returning grad log p(h).
    hNew <- h
    pNew <- p
    pNew <- pNew + 0.5*eps*grad_logpost(hNew)
    for (i in 1:L) {
        hNew <- hNew + eps*pNew
        if (i < L) {
            pNew <- pNew + eps*grad_logpost(hNew)
        } else {
            pNew <- pNew + 0.5*eps*grad_logpost(hNew)
        }
    }
    pNew <- -pNew
    list(hNew = hNew, pNew = pNew)
}

# storage
store_theta <- matrix(0, nsim, 3)   # [mu alp sigh2]
store_h <- matrix(0, nsim, T)
for (isim in 1:(nsim+burnin)) {
    # sample gam = (mu, alp)'
    X <- cbind(rep(1, T), exp(h))
    iSy <- 1/exp(h)         # diagonal of Sigma_y^{-1}
    XiSy <- t(X*iSy)        # X'*Sigma_y^{-1}, a dense 2-by-T block
    Kgam <- iVgam + XiSy %*% X
    gam_hat <- solve(Kgam, as.numeric(iVgam %*% gam0) + as.numeric(XiSy %*% y))
    gam <- as.numeric(gam_hat + solve(chol(Kgam), rnorm(2)))
    mu <- gam[1]; alp <- gam[2]

    # sample h using HMC
    logpost <- function(ht) logpost_svm_h(y, mu, alp, ht, h0, sigh2, HH)
    grad <- function(ht) grad_logpost_svm_h(y, mu, alp, ht, h0, sigh2, HH)
    p0 <- rnorm(T)   # initial momentum
    lf <- leapfrog(h, p0, eps, L, grad)
    hc <- lf$hNew; pc <- lf$pNew
    H0 <- -logpost(h)  + 0.5*sum(p0*p0)
    Hc <- -logpost(hc) + 0.5*sum(pc*pc)
    if (log(runif(1)) < -(Hc - H0)) {
        h <- hc
        if (isim > burnin)
            accept_h <- accept_h + 1
    }

    # # sample h using a Laplace-based ARMH step
    # out <- sample_SVM_h_ARMH(y, alp, mu, h, h0, sigh2, HH)
    # h <- out$h
    # if (isim > burnin)
    #     accept_h <- accept_h + out$accept

    # sample sigh2
    # R's rgamma takes shape and scale
    sigh2 <- 1/rgamma(1, shape = nu_h + T/2,
        scale = 1/(S_h + as.numeric(crossprod(h-h0, HH %*% (h-h0)))/2))

    # sample h0
    Kh0 <- 1/b0 + 1/sigh2
    h0_hat <- (a0/b0 + h[1]/sigh2)/Kh0
    h0 <- h0_hat + rnorm(1)/sqrt(Kh0)

    if (isim > burnin) {
        isave <- isim - burnin
        store_h[isave, ]     <- h
        store_theta[isave, ] <- c(mu, alp, sigh2)
    }
}
cat(sprintf("HMC acceptance rate = %.3f\n", accept_h/nsim))
# posterior summaries
h_mean <- colMeans(exp(store_h/2))
h_CI <- apply(exp(store_h/2), 2, quantile, probs = c(0.05, 0.95),
              type = 5)   # 90% credible interval
h_lower <- h_CI[1, ]
h_upper <- h_CI[2, ]

theta_mean <- colMeans(store_theta)
theta_CI <- apply(store_theta, 2, quantile, probs = c(0.025, 0.975),
                  type = 5)   # 95% credible interval

cat("posterior means of [mu, alp, sigh2]:", theta_mean, "\n")
cat("95% credible intervals (rows: 2.5%, 97.5%):\n")
print(theta_CI)

# Plot posterior mean and 90% credible interval for exp(h_t/2)
tt <- seq(2013, 2016, length.out = T)

setEPS()
postscript("SVM_h.eps", width = 9, height = 3.5)
par(mar = c(4, 4, 1, 1), cex.axis = 1.2, cex.lab = 1.2)
plot(tt, h_mean, type = "n", xlim = c(2013, 2016),
     ylim = range(c(h_lower, h_upper)), xaxt = "n", bty = "n",
     xlab = "", ylab = "")
polygon(c(tt, rev(tt)), c(h_lower, rev(h_upper)), col = gray(0.8), border = NA)
lines(tt, h_mean, col = "black", lwd = 1.5)
axis(1, at = c(2013, 2014, 2015, 2016))   # <-- only these ticks
dev.off()
