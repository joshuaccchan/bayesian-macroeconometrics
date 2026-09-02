# linreg_tvp.R
# Gibbs sampler for a time-varying parameter (TVP) regression, applied to a
# Phillips curve for US PCE inflation. The model is
#   y_t    = x_t'*beta_t + eps_t,     eps_t ~ N(0, sig2),
#   beta_t = beta_{t-1} + eta_t,      eta_t ~ N(0, Omega),
# with x_t = (1, gap_t, y_{t-1})' and Omega = diag(omega_1^2,...,omega_k^2), so
# each coefficient follows an independent random walk.
# The stacked coefficient path beta is drawn in one block from its Gaussian
# full conditional, whose block-tridiagonal precision matrix uses the
# Kronecker structure (H'H) kron Omega^{-1}; the remaining blocks
# (sig2, omega_j^2, beta0) use standard conjugate updates.
# The stacked design matrix Z = diag(x_1',...,x_T') is formed by SURform.R.
#
# Requires: SURform.R

suppressMessages(library(Matrix))
source("SURform.R")

set.seed(42)
nsim <- 20000; burnin <- 1000

# load data (columns PCECTPI and Output Gap, first 240 rows: 1960Q1-2019Q4)
data <- as.matrix(read.csv("USPCE_OutputGap.csv")[1:240, c("PCECTPI", "Output.Gap")])
infl <- data[,1]   # PCE inflation
gap  <- data[,2]   # output gap

# construct y and X
y <- infl[2:length(infl)]
g <- gap[2:length(gap)]
ylag <- infl[1:(length(infl)-1)]   # y_{t-1}
T <- length(y)
X <- cbind(rep(1,T), g, ylag)
k <- ncol(X)

# prior hyperparameters
beta00 <- numeric(k)
iVbeta0 <- 1/100*diag(k)
nu_sig <- 3;  S_sig <- 1   # IG prior for sigma^2
nu_om <- 3                 # IG prior for omega_j^2
S_om  <- c(0.125, 0.025, 0.025)^2*(nu_om-1)

# initialize chain
beta0 <- numeric(k)
beta_ols <- solve(crossprod(X), crossprod(X, y))
sig2 <- mean((y - X %*% beta_ols)^2)
omega2 <- 0.01^2 * rep(1,k)

# precompute a few things
S1 <- sparseMatrix(i = 2:T, j = 1:(T-1), x = 1, dims = c(T,T))
H  <- Diagonal(T) - S1
HH <- crossprod(H)
Z <- SURform(X)
ZZ <- crossprod(Z)
Zy <- as.numeric(crossprod(Z, y))

# storage
store_beta <- matrix(0, nsim, T*k)
    # [beta0', sig2, omega2']
store_theta <- matrix(0, nsim, 2*k + 1)

for (isim in 1:(nsim + burnin)) {
    # sample beta
    iOmega <- Diagonal(x = 1/omega2)
    P <- kronecker(HH, iOmega)   # prior precision
    Kbeta <- P + ZZ/sig2
    # Matrix's chol() returns the UPPER factor with no fill-reducing
    # permutation, so it is MATLAB's chol(Kbeta,'lower')' directly
    CKbeta <- chol(Kbeta)
    beta_hat <- solve(Kbeta, as.numeric(P %*% rep(beta0,T)) + Zy/sig2)
    beta <- as.numeric(beta_hat + solve(CKbeta, rnorm(k*T)))

    # sample sigma^2
    e <- y - as.numeric(Z %*% beta)
    # R's rgamma takes shape and scale
    sig2 <- 1/rgamma(1, shape = nu_sig + T/2, scale = 1/(S_sig + sum(e^2)/2))

    # sample omega_j^2
    Beta <- t(matrix(beta, k, T))   # row t is beta_t'
    SSE <- colSums((Beta - rbind(beta0, Beta[1:(T-1), , drop=FALSE]))^2)
    omega2 <- 1/rgamma(k, shape = nu_om + T/2, scale = 1/(S_om + 0.5*SSE))

    # sample beta0
    Kbeta0 <- iVbeta0 + diag(1/omega2)
    beta0_hat <- solve(Kbeta0, as.numeric(iVbeta0 %*% beta00)
        + beta[1:k]/omega2)
    Cbeta0 <- chol(Kbeta0)
    beta0 <- as.numeric(beta0_hat + solve(Cbeta0, rnorm(k)))

    if (isim > burnin) {
        isave <- isim - burnin
        store_beta[isave,] <- beta
        store_theta[isave,] <- c(beta0, sig2, omega2)
    }
}
Beta_mean <- t(matrix(colMeans(store_beta), k, T))
beta_q <- apply(store_beta, 2, quantile, probs = c(0.05, 0.95), type = 5)
Beta_q <- array(0, c(T, k, 2))   # Beta_q[,j,1] = 5%, Beta_q[,j,2] = 95%
Beta_q[,,1] <- t(matrix(beta_q[1,], k, T))
Beta_q[,,2] <- t(matrix(beta_q[2,], k, T))

theta_mean <- colMeans(store_theta)
cat("posterior means of [beta0', sig2, omega2']:\n")
print(theta_mean)

# plot coefficients with 90% CI
tt <- seq(1960.25, 2019.75, by = .25)

names <- c("Intercept", "Output gap", "Lagged inflation")

setEPS()
postscript("tvp_phillips_curve.eps", width = 6, height = 5)
par(mfrow = c(k,1), mar = c(2.5, 4.5, 1, 1))
for (j in 1:k) {
    lo <- Beta_q[,j,1]; hi <- Beta_q[,j,2]
    if (j == k) par(mar = c(4, 4.5, 1, 1))
    plot(tt, Beta_mean[,j], type = "n", ylim = range(c(lo, hi)),
         xlab = if (j == k) "Time" else "", ylab = names[j],
         cex.lab = 1.2, cex.axis = 1.2, bty = "l")
    polygon(c(tt, rev(tt)), c(lo, rev(hi)), col = gray(0.85), border = NA)

    lines(tt, Beta_mean[,j], col = "black", lwd = 1.5)
    box(bty = "l")
}
invisible(dev.off())
