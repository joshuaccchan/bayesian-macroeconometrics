# linreg_indep_predictive.R
# Gibbs sampler for an AR(2) model of US PCE inflation under the
# independent normal and inverse-gamma prior; constructs the one-step-ahead
# posterior predictive density for 2020Q1 by averaging conditional Gaussian
# forecasts across posterior draws.

set.seed(42)

nsim <- 20000; burnin <- 1000

# load data (column PCECTPI; rows 1960Q1-2019Q4)
data <- read.csv("USPCE.csv")$PCECTPI[1:240]
y0 <- data[1:4]    # initial conditions: [y_{-3},y_{-2},y_{-1},y_0]
y  <- data[5:240]  # sample used for estimation
T  <- length(y)

# regressors for AR(2): [1, y_{t-1}, y_{t-2}]
xlag1 <- c(y0[4], y[1:(T-1)])
xlag2 <- c(y0[3:4], y[1:(T-2)])
X <- unname(cbind(1, xlag1, xlag2))
k <- ncol(X)   # number of regressors

# prior hyperparameters (independent normal and inverse-gamma)
beta0  <- numeric(k)
iVbeta <- diag(k)/100   # prior precision of beta
nu0    <- 4
S0     <- 1

# one-step-ahead regressor vector (T+1)
xTp1  <- c(1, y[T], y[T-1])
y_obs <- 1.39
ygrid <- seq(-3, 8, length.out=300)

# initialize chain
beta <- as.numeric(solve(crossprod(X), crossprod(X, y)))
e    <- y - as.numeric(X %*% beta)
sig2 <- sum(e^2)/T

store_theta <- matrix(0, nsim, k+1)    # [beta' sig2]
store_fy    <- numeric(length(ygrid))  # store only the sum

# Gibbs sampler
for (isim in 1:(nsim + burnin)) {
    # sample beta
    Dbeta <- solve(iVbeta + crossprod(X)/sig2)
    beta_hat <- as.numeric(Dbeta %*% (iVbeta %*% beta0 + crossprod(X, y)/sig2))
    C <- t(chol(Dbeta))   # R's chol() is upper triangular; transpose for lower
    beta <- beta_hat + as.numeric(C %*% rnorm(k))

    # sample sig2
    # R's rgamma takes shape and scale
    e <- y - as.numeric(X %*% beta)
    sig2 <- 1/rgamma(1, shape = nu0 + T/2, scale = 1/(S0 + 0.5*sum(e^2)))

    if (isim > burnin) {
        isave <- isim - burnin
        store_theta[isave, ] <- c(beta, sig2)

        # store one-step-ahead predictive density
        mu1  <- sum(xTp1*beta)
        s2_1 <- sig2
        fy <- dnorm(ygrid, mu1, sqrt(s2_1))
        store_fy <- store_fy + fy
    }
}

# posterior summaries
theta_mean <- colMeans(store_theta)
theta_CI   <- apply(store_theta, 2, quantile, probs=c(.025, .975), type=5)

cat("Posterior mean of [beta' sig2]:\n")
print(theta_mean)
cat("95% posterior intervals (rows: 2.5%, 97.5%):\n")
print(theta_CI)

# Plot predictive density
fy_hat <- store_fy/nsim
plot(ygrid, fy_hat, type="l", col="black", lwd=2, bty="l",
     xlab=expression(y[T+1]),
     ylab=expression(p(y[T+1] ~ "|" ~ bold(y), bold(x)[T+1])))
abline(v=y_obs, lty=2, lwd=2)
