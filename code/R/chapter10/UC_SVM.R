# UC_SVM.R
# Metropolis-within-Gibbs sampler for the unobserved components stochastic
# volatility in mean model with time-varying coefficients, fitted to quarterly
# PCE inflation. The measurement equation is
#   y_t = tau_t + alp_t*exp(h_t) + eps_t,   eps_t ~ N(0, exp(h_t)),
# where the coefficient vector gam_t = (tau_t, alp_t)' follows a random walk and
# the log-volatility h_t is a random walk. The coefficient path gam and the
# static parameters use conjugate updates; h is sampled by the Laplace-based
# acceptance-rejection MH step (sample_SVM_h_ARMH.R).
# The stacked design matrix is built with SURform.R.
#
# Requires: sample_SVM_h_ARMH.R, SURform.R, SV_RW_gaussian_approx.R

suppressMessages(library(Matrix))
source("sample_SVM_h_ARMH.R")
source("SURform.R")
source("SV_RW_gaussian_approx.R")

set.seed(42)   # for reproducibility
nsim <- 20000; burnin <- 5000

# load PCE data - 1960Q1-2024Q4 (column PCECTPI, 260 rows)
data <- read.csv("USPCE.csv")$PCECTPI
y <- data
T <- length(y)

# prior hyperparameters
agam <- c(2, 0); iVgam <- diag(2)/100
nuOmega <- 3;  SOmega <- (nuOmega-1)*c(0.25^2, 0.10^2)
nuh <- 3; Sh <- 0.2^2*(nuh - 1)
ah <- 0; Vh <- 100

# initialize
omega <- c(0.25^2, 0.10^2)   # store the diagonal elements
sigh2 <- 0.2^2
h0 <- log(var(y))
tau <- mean(y)*rep(1, T)
h <- SV_RW_gaussian_approx((y - tau)^2, h0, sigh2)
exp_h <- exp(h)
gam0 <- numeric(2)

# storage
store_theta <- matrix(0, nsim, 6)   # [h0 sigh2 omega' gam0']
store_tau <- matrix(0, nsim, T)
store_alp <- matrix(0, nsim, T)
store_h <- matrix(0, nsim, T)

# precompute fixed matrices
S1gam <- sparseMatrix(i = 3:(2*T), j = 1:(2*(T-1)), x = rep(1, 2*(T-1)),
                      dims = c(2*T, 2*T))
Hgam <- Diagonal(2*T) - S1gam
S1 <- sparseMatrix(i = 2:T, j = 1:(T-1), x = 1, dims = c(T, T))
H  <- Diagonal(T) - S1
HH <- crossprod(H)
accept_h <- 0

for (isim in 1:(nsim + burnin)) {
    # sample gam
    Xgam <- SURform(cbind(rep(1, T), exp_h))
    tmp <- t(Xgam) %*% Diagonal(x = 1/exp_h)
        # prior precision
    Pgam <- t(Hgam) %*% kronecker(Diagonal(T), Diagonal(x = 1/omega)) %*% Hgam
    Kgam <- forceSymmetric(Pgam + tmp %*% Xgam)
    Cgam <- chol(Kgam)   # upper factor: Kgam = t(Cgam) %*% Cgam
    gamhat <- solve(Kgam,
                    as.numeric(Pgam %*% as.vector(kronecker(rep(1, T), gam0))) +
                    as.numeric(tmp %*% y))
    gam <- as.numeric(gamhat + solve(Cgam, rnorm(2*T)))
    tau <- gam[seq(1, 2*T, by = 2)]
    alp <- gam[seq(2, 2*T, by = 2)]

    # sample gam0
    Kgam0 <- iVgam + diag(1/omega)
    gam0hat <- solve(Kgam0, as.numeric(iVgam %*% agam) + gam[1:2]/omega)
    gam0 <- as.numeric(gam0hat + solve(chol(Kgam0), rnorm(2)))

    # sample h using acceptance-rejection MH
    out <- sample_SVM_h_ARMH(y, alp, tau, h, h0, sigh2, HH)
    h <- out$h; accept <- out$accept
    exp_h <- exp(h)
    if (isim > burnin)
        accept_h <- accept_h + accept

    # sample Omega
    e_gam <- t(matrix(gam - c(gam0, gam[1:(2*T-2)]), 2, T))
    newSOmega <- SOmega + colSums(e_gam^2)/2
    # R's rgamma takes shape and scale
    omega <- 1/rgamma(2, shape = nuOmega + T/2, scale = 1/newSOmega)

    # sample sigh2
    e_h <- c(h[1]-h0, diff(h))
    newSh <- Sh + sum(e_h^2)/2
    sigh2 <- 1/rgamma(1, shape = nuh + T/2, scale = 1/newSh)

    # sample h0
    Kh0 <- 1/Vh + 1/sigh2
    h0hat <- (ah/Vh + h[1]/sigh2)/Kh0
    h0 <- h0hat + rnorm(1)/sqrt(Kh0)

    # store draws
    if (isim > burnin) {
        isave <- isim - burnin
        store_tau[isave, ] <- tau
        store_alp[isave, ] <- alp
        store_h[isave, ]   <- h
        store_theta[isave, ] <- c(h0, sigh2, omega, gam0)
    }
}
cat(sprintf("Acceptance rate for h = %.3f\n", accept_h/nsim))
# posterior summaries
tau_mean <- colMeans(store_tau)
tau_CI   <- apply(store_tau, 2, quantile, probs = c(0.05, 0.95), type = 5)
tau_lower <- tau_CI[1, ]
tau_upper <- tau_CI[2, ]

alp_mean <- colMeans(store_alp)
alp_CI   <- apply(store_alp, 2, quantile, probs = c(0.05, 0.95), type = 5)
alp_lower <- alp_CI[1, ]
alp_upper <- alp_CI[2, ]

theta_mean <- colMeans(store_theta)
cat("posterior means of [h0, sigh2, omega_tau2, omega_alp2, tau00, alp00]:\n")
print(theta_mean)

# plot alpha_t with 90% CI
tt <- seq(1960, 2024.75, length.out = T)
setEPS()
postscript("UC_SVM.eps", width = 9, height = 3)
par(mar = c(4, 4, 1, 1), cex.axis = 1.2, cex.lab = 1.2)
plot(tt, alp_mean, type = "n", xlim = c(min(tt)-1, max(tt)+1),
     ylim = range(c(alp_lower, alp_upper)), bty = "n", xlab = "", ylab = "")
polygon(c(tt, rev(tt)), c(alp_lower, rev(alp_upper)), col = gray(0.8),
        border = NA)
lines(tt, alp_mean, col = "black", lwd = 1.5)
dev.off()
