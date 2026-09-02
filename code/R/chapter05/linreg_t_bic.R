# linreg_t_bic.R
# Computes the BIC for the AR(2) model with Student-t errors for US
# PCE inflation. The log-likelihood is stored at each post-burn-in
# Gibbs iteration, and the BIC is computed using the maximum sampled
# log-likelihood as a stand-in for the maximum likelihood value.

suppressMessages(library(Matrix))
source("sample_nu_griddy.R")

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

# prior hyperparameters (independent normal and inverse-gamma)
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
iLam <- Diagonal(x = 1/lam)  # sparse version of Lambda^{-1}

store_theta <- matrix(0, nsim, k+2)  # [beta', sig2, nu]
store_lam <- matrix(0, nsim, T)
store_llike <- numeric(nsim)
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
        llike <- T*(lgamma((nu+1)/2) - lgamma(nu/2) - .5*log(nu*pi*sig2)) -
            (nu+1)/2*sum(log(1 + as.numeric(y - X %*% beta)^2/(sig2*nu)))
        store_llike[isave] <- llike
    }
}
# Posterior summaries
theta_mean <- colMeans(store_theta)
theta_lo <- apply(store_theta, 2, quantile, probs = .025, type = 5)
theta_hi <- apply(store_theta, 2, quantile, probs = .975, type = 5)

lam_mean <- colMeans(store_lam)
lam_lo <- apply(store_lam, 2, quantile, probs = 0.025, type = 5)
lam_hi <- apply(store_lam, 2, quantile, probs = 0.975, type = 5)

max_llike <- max(store_llike)
BIC <- -2*max_llike + 5*log(T)
cat(sprintf("BIC: %.2f\n", BIC))
