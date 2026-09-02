# linreg_t.R
# Gibbs sampler for an AR(2) model of US PCE inflation with Student-t
# errors. The model is
#   y_t = beta_1 + beta_2 y_{t-1} + beta_3 y_{t-2} + eps_t,
#   (eps_t | lam_t) ~ N(0, sig2*lam_t),  lam_t ~ IG(nu/2, nu/2),
# with independent normal-inverse-gamma priors on (beta, sig2) and a
# uniform prior on nu over (2, nu_ub). The block for nu uses a
# Griddy-Gibbs step via sample_nu_griddy.R.

suppressMessages(library(Matrix))
source("sample_nu_griddy.R")
source("shaded_band.R")

set.seed(42)
nsim <- 20000; burnin <- 1000

# load data (MATLAB Range B2:B241 = first 240 rows: 1960Q1-2019Q4)
data <- read.csv("USPCE.csv")[1:240, 2]
y0 <- data[1:4]            # initial conditions: [y_{-3},y_{-2},y_{-1},y_0]
y <- data[5:length(data)]  # sample used for estimation
T <- length(y)

# regressors for AR(2): [1, y_{t-1}, y_{t-2}]
xlag1 <- c(y0[4], y[1:(T-1)])
xlag2 <- c(y0[3:4], y[1:(T-2)])
X <- unname(cbind(rep(1, T), xlag1, xlag2))
k <- ncol(X)     # number of regressors

# prior hyperparameters (indep normal and inverse-gamma)
# (the .m stores iVbeta as speye(k); with k = 3 a dense identity is
#  both simpler and faster in R)
beta0  <- numeric(k)
iVbeta <- diag(k)/100      # prior precision of beta
nu0    <- 4
S0     <- 1
nu_ub  <- 50               # prior upperbound of nu

# initialize the Markov chain
nu <- 5
beta <- as.numeric(solve(crossprod(X), crossprod(X, y)))
sig2 <- sum((y - X %*% beta)^2)/T
lam <- rep(1, T)
iLam <- Diagonal(x = 1/lam)

store_theta <- matrix(0, nsim, k+2)  # [beta', sig2, nu]
store_lam <- matrix(0, nsim, T)
n_grid <- 500

for (isim in 1:(nsim + burnin)) {
    # sample beta
    Dbeta <- solve(iVbeta + as.matrix(crossprod(X, iLam %*% X))/sig2)
    beta_hat <- as.numeric(Dbeta %*% (iVbeta %*% beta0
        + as.numeric(crossprod(X, iLam %*% y))/sig2))
    C <- t(chol(Dbeta))    # R's chol() is upper, so transpose
    beta <- beta_hat + as.numeric(C %*% rnorm(k))

    # sample sig2
    e <- as.numeric(y - X %*% beta)
    # R's rgamma takes shape and scale
    sig2 <- 1/rgamma(1, shape = nu0 + T/2,
        scale = 1/(S0 + as.numeric(crossprod(e, iLam %*% e))/2))

    # sample lam
    lam <- 1/rgamma(T, shape = (nu+1)/2, scale = 2/(nu + e^2/sig2))
    iLam <- Diagonal(x = 1/lam)

    # sample nu
    nu <- sample_nu_griddy(lam, nu_ub, n_grid)

        # store the parameters
    if (isim > burnin) {
        isave <- isim - burnin
        store_theta[isave, ] <- c(beta, sig2, nu)
        store_lam[isave, ] <- lam
    }
}
# Posterior summaries
theta_mean <- colMeans(store_theta)
theta_lo <- apply(store_theta, 2, quantile, probs = .025, type = 5)
theta_hi <- apply(store_theta, 2, quantile, probs = .975, type = 5)

lam_mean <- colMeans(store_lam)
lam_lo <- apply(store_lam, 2, quantile, probs = 0.025, type = 5)
lam_hi <- apply(store_lam, 2, quantile, probs = 0.975, type = 5)

cat("posterior means [beta_1, beta_2, beta_3, sig2, nu]:\n")
print(round(theta_mean, 3))

par(mfrow = c(1, 2), mar = c(4.5, 4.5, 1, 1))

# left panel: posterior of nu
hist(store_theta[, ncol(store_theta)], breaks = 30,
     col = gray(0.8), border = "black",
     main = "", xlab = expression(nu), ylab = "Frequency")

# right panel: lambda_t (log scale)
    # time index: 1961Q1 to 2019Q4
tq <- seq(as.Date("1961-01-01"), by = "3 months", length.out = T)
plot(tq, lam_mean, type = "n", log = "y",
     xlim = c(as.Date("1960-01-01"), as.Date("2020-01-01")),
     ylim = range(c(lam_lo, lam_hi)),
     xlab = "", ylab = expression(paste(lambda[t], " (log scale)")))
shaded_band(tq, lam_lo, lam_hi, 0.85)
lines(tq, lam_mean, col = "black", lwd = 2)
box()
