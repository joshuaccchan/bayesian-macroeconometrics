# linreg_bqr.R
# Gibbs sampler for Bayesian quantile regression of growth-at-risk.
# The model is
#   y_{t+h} = beta_1 + beta_2 gdp_t + beta_3 nfci_t + eps_{t+h},
#   eps_{t+h} ~ AL(0, sig2, tau),
# with the location-scale mixture-of-normals representation
# (eps | lam) ~ N(vartheta*lam, varphi*sig2*lam), lam ~ G(1, 1/sig2),
# and independent N + IG priors on (beta, sig2). The latent scales
# {lam_t} are updated via the inverse Gaussian representation; see
# igaussrnd.R.

source("igaussrnd.R")

set.seed(42)

nsim <- 20000; burnin <- 1000
tau <- 0.05   # quantile level (e.g., 0.05, 0.10, 0.50)
h <- 4        # forecast horizon in quarters (e.g., 1 or 4)

# AL mixture constants
vartheta <- (1 - 2*tau) / (tau*(1 - tau))
varphi <- 2 / (tau*(1 - tau))

# load GDP-NFCI merged data
#   Date is a quarter-start date
#   GDP is quarterly (annualized) real GDP growth
#   NFCI is quarterly average of weekly NFCI
TBL <- read.csv("GDP_NFCI_merged.csv")
TBL$Date <- as.Date(TBL$Date, format = "%m/%d/%Y")

# keep observations with NFCI available
idx <- !is.na(TBL$NFCI) & !is.na(TBL$GDP)
TBL <- TBL[idx, ]
gdp  <- TBL$GDP
nfci <- TBL$NFCI

# Construct regression for Growth-at-Risk:
#   y_{t+h} = beta0 + beta1*gdp_t + beta2*nfci_t + eps_{t+h}
T0 <- length(gdp)
T  <- T0 - h
y <- gdp[(1+h):T0]   # y_{t+h}
X <- cbind(rep(1, T), gdp[1:T], nfci[1:T])
k <- ncol(X)

# prior hyperparameters (indep normal and inverse-gamma)
beta0 <- numeric(k)
iVbeta <- diag(k)/100   # prior precision
nu0 <- 3
S0 <- 1*(nu0 - 1)

# initialize the Markov chain
beta <- as.numeric(solve(crossprod(X), crossprod(X, y)))
sig2 <- mean((y - X %*% beta)^2)
# R's rgamma takes shape and scale
lam <- rgamma(T, shape = 1, scale = sig2)
ilam <- 1/lam    # iLam is diagonal; keep 1/lam as a vector
store_theta <- matrix(0, nsim, k+1)   # [beta' sig2]

# Gibbs sampler starts here
for (isim in 1:(nsim + burnin)) {

    # sample beta
    ytilde <- y - vartheta*lam
    Dbeta <- solve(iVbeta + crossprod(X, ilam*X)/(varphi*sig2))
    beta_hat <- Dbeta %*% (iVbeta %*% beta0
                           + crossprod(X, ilam*ytilde)/(varphi*sig2))
    C <- t(chol(Dbeta))
    beta <- as.numeric(beta_hat + C %*% rnorm(k))

    # sample sig2
    e <- as.numeric(y - X %*% beta - vartheta*lam)
    S_hat <- S0 + sum(lam) + sum(e*ilam*e)/(2*varphi)
    sig2  <- 1/rgamma(1, shape = nu0 + 3*T/2, scale = 1/S_hat)

    # sample lambda
    a_tau <- (vartheta^2 + 2*varphi)/(varphi*sig2)
    mu <- sqrt(vartheta^2 + 2*varphi) / abs(as.numeric(y - X %*% beta))
    lam <- 1/igaussrnd(a_tau*rep(1, T), mu)
    ilam <- 1/lam

    if (isim > burnin) {
        isave <- isim - burnin
        store_theta[isave, ] <- c(beta, sig2)
    }
}
theta_mean <- colMeans(store_theta)
theta_CI <- apply(store_theta, 2, quantile, probs = c(.025, .975), type = 5)

cat("Posterior mean (beta; sig2):\n")
print(theta_mean)
cat("Posterior 95% CI (rows: 2.5%, 97.5%):\n")
print(theta_CI)

# compute GaR_t(h; tau)
B <- store_theta[, 1:k]
Xgar <- X   # regressors at time t (aligned with y_{t+h})
GaR_draws <- Xgar %*% t(B)   # dimension is T x nsim
GaR_mean <- rowMeans(GaR_draws)

# Credible bands (pointwise)
GaR_lo95 <- apply(GaR_draws, 1, quantile, probs = 0.025, type = 5)
GaR_hi95 <- apply(GaR_draws, 1, quantile, probs = 0.975, type = 5)

# align dates with forecasted outcome y_{t+h}
date_f <- TBL$Date[(1+h):T0]

# Start two quarters before first forecast date
x_start <- seq(date_f[1], by = "-6 months", length.out = 2)[2]
x_end   <- date_f[length(date_f)]

plot(date_f, GaR_mean, type = "l", col = "black", lwd = 2, bty = "n",
     xlim = c(x_start, x_end), xlab = "", ylab = "GaR")
