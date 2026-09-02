# linreg_ssvs.R
# Collapsed Gibbs sampler for the point-mass spike-and-slab SSVS
# regression of US PCE inflation on an intercept and the first two
# lags of 16 macroeconomic and financial indicators.
# Requires logpost_gam.R and PCE_regression_data.csv.

source("logpost_gam.R")

set.seed(42)

nsim <- 20000
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

    # priors
iVbeta <- diag(k)/100
nu0 <- 3; S0 <- 1
bp <- .5*rep(1, k)  # prior inclusion probabilities

store_gam <- matrix(0, nsim, k)
store_betatilde <- matrix(0, nsim, k)

    # initialize the Markov chain
beta_ols <- as.numeric(solve(crossprod(X), crossprod(X, y)))
sig2_ols <- sum((y - X %*% beta_ols)^2)/T
beta_ols_std <- sqrt(sig2_ols*diag(solve(crossprod(X))))
beta <- beta_ols
sig2 <- sig2_ols
gam <- as.numeric(abs(beta)/beta_ols_std > 1.65)
gam <- c(1, gam[2:k])   # fix gamma_1 = 1

for (isim in 1:(nsim + burnin)) {
    # sample gamma marginal of beta (single-site)
    for (j in 2:k) {
        # evaluate log-kernel at gamma_j = 0
        gam0 <- gam
        gam0[j] <- 0
        l0 <- logpost_gam(gam0, y, X, sig2, bp, iVbeta)

        # evaluate log-kernel at gamma_j = 1
        gam1 <- gam
        gam1[j] <- 1
        l1 <- logpost_gam(gam1, y, X, sig2, bp, iVbeta)

        # stable Bernoulli draw
        mm <- max(l0, l1)
        p1 <- exp(l1 - mm)/(exp(l0 - mm) + exp(l1 - mm))
        gam[j] <- as.numeric(runif(1) < p1)
    }

    # sample beta | gamma, sig2
    Xtilde <- sweep(X, 2, gam, "*")   # X %*% diag(gam)
    Dbeta <- solve(iVbeta + crossprod(Xtilde)/sig2)
    beta_hat <- as.numeric(Dbeta %*% (crossprod(Xtilde, y)/sig2))
    C <- t(chol(Dbeta))
    beta <- beta_hat + as.numeric(C %*% rnorm(k))

    # sample sig2
    e <- as.numeric(y - Xtilde %*% beta)
    sig2 <- 1/rgamma(1, shape = nu0 + T/2, scale = 1/(S0 + sum(e^2)/2))

    if (isim > burnin) {
        # store the parameters
        isave <- isim - burnin
        store_betatilde[isave, ] <- beta*gam
        store_gam[isave, ] <- gam
    }

}
betatilde_mean <- colMeans(store_betatilde[, 2:k])
betatilde_ci <- t(apply(store_betatilde[, 2:k], 2, quantile,
                        probs = c(.05, .95), type = 5))
gam_mean <- colMeans(store_gam)
    # model size - exclude the intercept indicator
sumgam_mean <- mean(rowSums(store_gam[, 2:k]))

gray_col <- gray(0.5)
short_names <- c("PCE inflation", "Oil price", "FFR", "10y yield",
    "Term spread", "AAA-FFR spread", "Real M2", "Consumer credit",
    "UM sentiment", "Cap. utilization", "Real GDP", "Real PCE",
    "Ind. production", "Unemployment", "Payrolls", "Housing starts")
ytick_labels <- c(paste0(short_names, " (lag 1)"),
                  paste0(short_names, " (lag 2)"))

# posterior summaries
cat(sprintf("posterior mean model size (excl. intercept): %.2f\n", sumgam_mean))
cat("predictors with posterior inclusion probability > 0.5:\n")
for (idx in which(gam_mean[2:k] > 0.5)) {
    cat(sprintf("  %-28s P(gam=1|y) = %.3f  E(betatilde|y) = % .3f\n",
                ytick_labels[idx], gam_mean[idx+1], betatilde_mean[idx]))
}

op <- par(mar = c(4, 10, 1, 1))
plot(NULL, xlim = range(betatilde_ci), ylim = c(0, k),
     xlab = expression(tilde(beta)[j]), ylab = "", yaxt = "n", bty = "n")
    # 0-line
lines(c(0, 0), c(0, k), col = gray_col, lwd = 1)
    # Credible intervals + means
for (idx in 1:(k-1)) {
    lines(c(betatilde_ci[idx, 1], betatilde_ci[idx, 2]), c(idx, idx),
          col = gray_col, lwd = 2)
    points(betatilde_mean[idx], idx, pch = 19, col = gray_col)
}
axis(2, at = 1:(k-1), labels = ytick_labels, las = 1, cex.axis = 0.6)
par(op)
