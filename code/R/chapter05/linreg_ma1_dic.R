# linreg_ma1_dic.R
# Computes the DIC for the AR(2) model with MA(1) errors (ARMA(2,1))
# for US PCE inflation. The deviance is stored at each post-burn-in
# Metropolis-within-Gibbs iteration; the DIC equals the posterior
# mean deviance plus the effective number of parameters.

suppressMessages(library(Matrix))
source("loglike_MA1.R")
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
store_dev <- numeric(nsim)
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
        store_dev[isave] <- -2*loglike_MA1(psi, y - X %*% beta, sig2)
    }
}

# acceptance rate
accept_rate <- count_psi/nsim
cat(sprintf("Acceptance rate for psi: %.2f\n", accept_rate))

theta_hat <- colMeans(store_theta)
beta_hat <- theta_hat[1:3]
sig2_hat <- theta_hat[4]
psi_hat <- theta_hat[5]
pD <- mean(store_dev) + 2*loglike_MA1(psi_hat, y - X %*% beta_hat, sig2_hat)
DIC <- mean(store_dev) + pD
cat(sprintf("DIC: %.2f\n", DIC))
cat(sprintf("pD: %.1f\n", pD))
