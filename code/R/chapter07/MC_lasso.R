# MC_lasso.R
# Monte Carlo benchmark for the Bayesian Lasso of Example 7.2:
# generates R = 100 datasets of size T = 100 with k regressors and
# true coefficient vector (1, 1, 0, ..., 0)', then compares the
# posterior-mean MSE of the Bayesian Lasso against ordinary least
# squares and ridge regression. The shrinkage parameter lambda is
# set as lambda = k * sqrt(sig2_LS) / sum |beta_LS|. Requires
# fit_BayesLasso.R, fit_BayesRidge.R, and igaussrnd.R.

source("fit_BayesRidge.R")
source("fit_BayesLasso.R")

set.seed(42)

nsim <- 5000
burnin <- 1000
R <- 100
T <- 100
k <- 20  # number of regressors (change to 50 for the high-dim case)

# priors
nu0 <- 5; S0 <- 1

# true parameter values
truebeta <- c(1, 1, numeric(k-2))
truesig2 <- 1

store_MSE <- matrix(0, R, 3)
for (idata in 1:R) {   # a plain loop; the parallel package could parallelize it
    # generate data
    X <- matrix(5*runif(T*k), T, k)
    y <- as.numeric(X %*% truebeta) + sqrt(truesig2)*rnorm(T)

    beta_ols <- as.numeric(solve(crossprod(X), crossprod(X, y)))
    sig2_ols <- sum((y - X %*% beta_ols)^2)/T
    lambda <- k*sqrt(sig2_ols)/sum(abs(beta_ols))

    beta_ridge <- fit_BayesRidge(y, X, 2*k*sig2_ols/sum(beta_ols^2))
    beta_lasso <- fit_BayesLasso(y, X, lambda, S0, nu0, nsim, burnin)$beta_mean

    store_MSE[idata, ] <- c(mean((truebeta - beta_ols)^2),
                            mean((truebeta - beta_ridge)^2),
                            mean((truebeta - beta_lasso)^2))
}

MSE_med <- apply(store_MSE, 2, median)
cat(sprintf("\nMedian MSE  (k = %d):\n", k))
cat(sprintf("  OLS    %.4f\n  Ridge  %.4f\n  Lasso  %.4f\n",
            MSE_med[1], MSE_med[2], MSE_med[3]))

boxplot(store_MSE, names = c("LS", "Ridge", "Lasso"), range = 10,
        border = "black", pch = 3, frame.plot = FALSE)
