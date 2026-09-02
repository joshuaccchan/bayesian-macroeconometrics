# linreg_ma1.R
# Metropolis-within-Gibbs sampler for an ARMA(2,1) model of US PCE
# inflation. The model is
#   y_t = beta_1 + beta_2 y_{t-1} + beta_3 y_{t-2} + eps_t,
#   eps_t = u_t + psi*u_{t-1},  u_t ~ N(0, sig2),
# with independent normal-inverse-gamma priors on (beta, sig2) and a
# uniform prior on psi over (-1, 1). The block for psi uses a
# random-walk Metropolis-Hastings step via sample_psi_RW.R.

suppressMessages(library(Matrix))
source("sample_psi_RW.R")

set.seed(42)
nsim <- 50000; burnin <- 1000

# load data (MATLAB Range B2:B241 = first 240 rows: 1960Q1-2019Q4)
data <- read.csv("USPCE.csv")[1:240, 2]
y0 <- data[1:4]                 # initial conditions
y <- data[5:length(data)]       # sample used for estimation
T <- length(y)

# regressors for AR(2): [1, y_{t-1}, y_{t-2}]
xlag1 <- c(y0[4], y[1:(T-1)])
xlag2 <- c(y0[3:4], y[1:(T-2)])
X <- unname(cbind(rep(1, T), xlag1, xlag2))
k <- ncol(X)     # number of regressors

# prior hyperparameters (indep normal and inverse-gamma)
# (the .m stores iVbeta as speye(k); with k = 3 a dense identity is
#  both simpler and faster in R)
beta0 <- numeric(k);  iVbeta <- diag(k)/100
nu0 <- 4;  S0 <- 1

# initialize the Markov chain
psi <- 0
beta <- as.numeric(solve(crossprod(X), crossprod(X, y)))
sig2 <- sum((y - X %*% beta)^2)/T

Hpsi <- bandSparse(T, k = c(0, -1),
                   diagonals = list(rep(1, T), rep(psi, T-1)))
store_theta <- matrix(0, nsim, k+2)  # [beta', sig2, psi]
count_psi <- 0
g_var <- 0.02  # proposal variance for RW-MH step for psi

for (isim in 1:(nsim + burnin)) {
    # sample beta
    X_tilde <- as.matrix(solve(Hpsi, X))
    y_tilde <- as.numeric(solve(Hpsi, y))
    Dbeta <- solve(iVbeta + crossprod(X_tilde)/sig2)
    beta_hat <- as.numeric(Dbeta %*% (iVbeta %*% beta0
        + crossprod(X_tilde, y_tilde)/sig2))
    beta <- beta_hat + as.numeric(t(chol(Dbeta)) %*% rnorm(k))

    # sample sig2
    e <- as.numeric(y - X %*% beta)
    u <- as.numeric(solve(Hpsi, e))
    # R's rgamma takes shape and scale
    sig2 <- 1/rgamma(1, shape = nu0 + T/2, scale = 1/(S0 + sum(u^2)/2))

    # sample psi
    out_psi <- sample_psi_RW(psi, e, sig2, g_var)
    psi <- out_psi$psi;  accept <- out_psi$accept
    Hpsi <- bandSparse(T, k = c(0, -1),
                       diagonals = list(rep(1, T), rep(psi, T-1)))

    # store the parameters
    if (isim > burnin) {
        isave <- isim - burnin
        count_psi <- count_psi + accept  # post-burnin
        store_theta[isave, ] <- c(beta, sig2, psi)
    }
}

# acceptance rate
accept_rate <- count_psi/nsim
cat(sprintf("Acceptance rate for psi: %.2f\n", accept_rate))

# Posterior summaries
theta_mean <- colMeans(store_theta)
theta_lo <- apply(store_theta, 2, quantile, probs = .025, type = 5)
theta_hi <- apply(store_theta, 2, quantile, probs = .975, type = 5)
cat("posterior means [beta_1, beta_2, beta_3, sig2, psi]:\n")
print(round(theta_mean, 3))

par(mfrow = c(1, 2), mar = c(4.5, 4.5, 1, 1))

# left panel: posterior of psi
hist(store_theta[, ncol(store_theta)], breaks = 30,
     col = gray(0.8), border = "black",
     main = "", xlab = expression(psi), ylab = "Frequency")

# right panel: trace plot of psi
plot(store_theta[, ncol(store_theta)], type = "l", col = "black", lwd = 1,
     xlim = c(0, nsim), xlab = "Iteration", ylab = expression(psi))
