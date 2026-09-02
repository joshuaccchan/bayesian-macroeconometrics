# linreg_NIG_predictive.R
# Closed-form Bayesian inference for an AR(2) model of US PCE inflation
# under the natural conjugate (normal-inverse-gamma) prior; computes the
# one-step-ahead Student-t posterior predictive density for 2020Q1 and
# reports the predictive percentile at the realized value.

source("tpdfLS.R")

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

# prior hyperparameters (normal-inverse-gamma)
beta0  <- numeric(k)   # prior mean
iVbeta <- diag(k)/100  # prior precision
nu0    <- 4            # prior degrees of freedom
S0     <- 1            # prior scale

# Posterior hyperparameters
Dbeta <- solve(iVbeta + crossprod(X))
beta_hat <- as.numeric(Dbeta %*% (iVbeta %*% beta0 + crossprod(X, y)))
S_hat <- S0 + as.numeric(crossprod(y) + beta0 %*% iVbeta %*% beta0
    - beta_hat %*% solve(Dbeta, beta_hat))/2
nu_hat <- nu0 + T/2

# Posterior predictive density for y_{T+1}
xTp1  <- c(1, y[T], y[T-1])         # regressor vector at T+1
ygrid <- seq(-3, 8, length.out=500) # evaluation grid
fy <- tpdfLS(ygrid, sum(xTp1*beta_hat),
    S_hat/nu_hat*(1 + as.numeric(xTp1 %*% Dbeta %*% xTp1)), 2*nu_hat)

# Plot predictive density
plot(ygrid, fy, type="l", col="black", lwd=2, bty="l",
     xlab=expression(y[T+1]),
     ylab=expression(p(y[T+1] ~ "|" ~ bold(y), bold(x)[T+1])))
abline(v=1.39, lty=2, lwd=2)

y_obs <- 1.39
z <- (y_obs - sum(xTp1*beta_hat))/
    sqrt(S_hat/nu_hat*(1 + as.numeric(xTp1 %*% Dbeta %*% xTp1)))
pct <- pt(z, 2*nu_hat)
cat("Posterior mean of beta:", beta_hat, "\n")
cat(sprintf("Predictive percentile at y_{T+1} = 1.39: %.2f%%\n", 100*pct))
