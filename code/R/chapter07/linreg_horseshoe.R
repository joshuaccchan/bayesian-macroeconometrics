# linreg_horseshoe.R
# Gibbs sampler for the regression of US PCE inflation on an
# intercept and the first two lags of 16 macroeconomic and financial
# indicators, with a horseshoe prior on the 32 slope coefficients
# and a diffuse normal prior on the intercept. The half-Cauchy local
# and global scales tau_j and theta are represented via the latent
# inverse-gamma scale mixture of Makalic and Schmidt (2016), yielding
# inverse-gamma full conditionals for (tau_j^2, theta^2, lam_tau,
# lam_theta). A small floor is imposed on tau_j^2 * theta^2 to avoid
# numerical ill-conditioning in the Gaussian update for beta.
# Requires PCE_regression_data.csv.

set.seed(42)

nsim <- 50000
burnin <- 1000

# Load data (columns B:Q, rows 2-241 of the CSV = 1960Q1-2019Q4)
data <- as.matrix(read.csv("PCE_regression_data.csv")[1:240, 2:17])
y_raw <- data[, 1]
X_raw <- data
p <- 2 # # of lags
T0 <- nrow(X_raw); m <- ncol(X_raw)
X <- matrix(0, T0, m*p) # construct predictors
for (j in 1:p) {
    X[, ((j-1)*m+1):(j*m)] <- rbind(matrix(NA_real_, j, m), X_raw[1:(T0-j), ])
}
    # Drop initial rows with missing values from lagging
y <- y_raw[(p+1):T0]
X <- X[(p+1):T0, ]
T <- length(y)
X <- cbind(rep(1, T), X) # add an intercept
k <- m*p + 1

# prior hyperparameters
Vbeta0 <- 100 # prior variance for the intercept
nu0 <- 3; S0 <- 1

store_beta <- matrix(0, nsim, k)
XtX <- crossprod(X)
Xty <- as.numeric(crossprod(X, y))

# initialize
# R's rgamma takes shape and scale, like MATLAB's gamrnd
beta_ols <- as.numeric(solve(XtX, Xty))
sig2_ols <- sum((y - X %*% beta_ols)^2)/T
beta <- beta_ols
sig2 <- sig2_ols
lam_tau <- 1/rgamma(k-1, shape = 1/2, scale = 1)
lam_theta <- 1/rgamma(1, shape = 1/2, scale = 1)
tau2 <- 1/rgamma(k-1, shape = 1/2, scale = lam_tau)
theta2 <- 1/rgamma(1, shape = 1/2, scale = lam_theta)

for (isim in 1:(nsim + burnin)) {
    # sample beta
    var_slope <- tau2 * theta2
    var_slope <- pmax(var_slope, 1e-10)  # set lower bounds
    prior_prec <- c(1/Vbeta0, 1/var_slope)
    Dbeta <- solve(diag(prior_prec, k) + XtX/sig2)
    beta_hat <- as.numeric(Dbeta %*% Xty)/sig2
    beta <- beta_hat + as.numeric(t(chol(Dbeta)) %*% rnorm(k))

    # sample sig2
    e <- as.numeric(y - X %*% beta)
    S_hat <- S0 + sum(e^2)/2
    sig2 <- 1/rgamma(1, shape = nu0 + T/2, scale = 1/S_hat)

    # sample tau2
    tmp1 <- 1/lam_tau + beta[2:k]^2/(2*theta2)
    tau2 <- 1/rgamma(k-1, shape = 1, scale = 1/tmp1)

    # sample theta2
    tmp2 <- 1/lam_theta + sum(beta[2:k]^2/tau2)/2
    theta2 <- 1/rgamma(1, shape = k/2, scale = 1/tmp2)

    # sample lam_tau, lam_theta
    lam_tau <- 1/rgamma(k-1, shape = 1, scale = 1/(1 + 1/tau2))
    lam_theta <- 1/rgamma(1, shape = 1, scale = 1/(1 + 1/theta2))

    # store the parameters
    if (isim > burnin) {
        isave <- isim - burnin
        store_beta[isave, ] <- beta
    }
}
beta_mean <- colMeans(store_beta[, 2:k])
beta_ci <- t(apply(store_beta[, 2:k], 2, quantile, probs = c(.05, .95),
                   type = 5))

gray_col <- gray(0.5)
short_names <- c("PCE inflation", "Oil price", "FFR", "10y yield",
    "Term spread", "AAA-FFR spread", "Real M2", "Consumer credit",
    "UM sentiment", "Cap. utilization", "Real GDP", "Real PCE",
    "Ind. production", "Unemployment", "Payrolls", "Housing starts")
ytick_labels <- c(paste0(short_names, " (lag 1)"),
                  paste0(short_names, " (lag 2)"))

# posterior summaries: the largest slope coefficients in absolute value
cat("largest posterior-mean slope coefficients (with 90% credible intervals):\n")
for (idx in order(abs(beta_mean), decreasing = TRUE)[1:5]) {
    cat(sprintf("  %-28s % .3f  [% .3f, % .3f]\n", ytick_labels[idx],
                beta_mean[idx], beta_ci[idx, 1], beta_ci[idx, 2]))
}

op <- par(mar = c(4, 10, 1, 1))
plot(NULL, xlim = range(beta_ci), ylim = c(0, k), xlab = expression(beta[j]),
     ylab = "", yaxt = "n", bty = "n")
    # 0-line
lines(c(0, 0), c(0, k), col = gray_col, lwd = 1)
    # Credible intervals + means
for (idx in 1:(k-1)) {
    lines(c(beta_ci[idx, 1], beta_ci[idx, 2]), c(idx, idx),
          col = gray_col, lwd = 2)
    points(beta_mean[idx], idx, pch = 19, col = gray_col)
}
axis(2, at = 1:(k-1), labels = ytick_labels, las = 1, cex.axis = 0.6)
par(op)
