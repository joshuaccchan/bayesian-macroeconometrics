# fit_BayesLasso.R
# Two-block Gibbs sampler for the Bayesian Lasso. The prior is the
# scale-mixture-of-normals representation of the Laplace prior:
#   (beta_j | sig2, tau_j^2) ~ N(0, sig2 * tau_j^2),
#   (tau_j^2 | lambda^2)     ~ G(1, lambda^2/2),
#   sig2 ~ IG(nu0, S0).
# In each iteration, (beta, sig2) are sampled jointly from their NIG
# full conditional, and each tau_j^2 is updated via the reciprocal
# of an inverse-Gaussian draw. Requires igaussrnd.R.
#
# Inputs:
#   y      : length-T response
#   X      : T-by-k design matrix
#   lambda : Lasso shrinkage hyperparameter
#   S0,nu0 : inverse-gamma hyperparameters for sig2
#   nsim   : number of post burn-in draws
#   burnin : number of burn-in iterations
#
# Outputs (returned in a list):
#   beta_mean : length-k posterior mean of beta
#   store_beta: nsim-by-k matrix of post burn-in beta draws

source("igaussrnd.R")

fit_BayesLasso <- function(y, X, lambda, S0, nu0, nsim, burnin) {
    T <- nrow(X); k <- ncol(X)
    XtX <- crossprod(X); Xty <- as.numeric(crossprod(X, y)); yy <- sum(y*y)
    store_beta <- matrix(0, nsim, k)

        # initialize
    # R's rgamma takes shape and scale, like MATLAB's gamrnd
    tau2 <- rgamma(k, shape = 1, scale = 2/lambda^2)
    for (isim in 1:(nsim + burnin)) {
            # sample beta and sigma^2
        Dbeta <- solve(diag(1/tau2, k) + XtX)
        beta_hat <- as.numeric(Dbeta %*% Xty)
        S_hat <- S0 + (yy - sum(beta_hat*solve(Dbeta, beta_hat)))/2
        sig2 <- 1/rgamma(1, shape = nu0 + T/2, scale = 1/S_hat)
        beta <- beta_hat +
            as.numeric(t(chol(sig2*Dbeta)) %*% rnorm(k))
            # sample tau2
        tmp <- lambda*sqrt(sig2)/abs(beta)
        tau2 <- 1/igaussrnd(lambda^2*rep(1, k), tmp)
            # store draws
        if (isim > burnin) {
            isave <- isim - burnin
            store_beta[isave, ] <- beta
        }
    }
    beta_mean <- colMeans(store_beta)
    list(beta_mean = beta_mean, store_beta = store_beta)
}
