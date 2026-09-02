# linreg_t_gd.R
# Computes the log marginal likelihood of the AR(2) model with
# Student-t errors for US PCE inflation via the modified harmonic mean
# (Geweke-Draper) estimator. The auxiliary density f is taken as a
# multivariate normal approximation to the posterior with covariance
# Q_theta, truncated to the (1-alpha)-quantile ellipsoid of the chi^2_m
# distribution to satisfy the thin-tail condition of Geweke (1999).

source("lmvnpdf.R")
source("ligampdf.R")

source("linreg_t.R")  # estimate the t model

nsim <- nrow(store_theta);  m <- ncol(store_theta)
T <- length(y)

# log prior
prior <- function(b, s, n) lmvnpdf(b, beta0, solve(iVbeta)) +
    ligampdf(s, nu0, S0) + log(1/(nu_ub - 2))

theta_hat <- colMeans(store_theta)
Qtheta <- cov(store_theta)
alp <- .05     # significance level for truncation
chi2q <- qchisq(1 - alp, m)

# Cholesky for stable logdet and quadratic forms
L <- t(chol(Qtheta))   # R's chol() is upper, so transpose
logdetQ <- 2*sum(log(diag(L)))

# log normalizing constant for f
const_f <- -0.5*m*log(2*pi) - 0.5*logdetQ - log(1 - alp)

store_w <- rep(-Inf, nsim)
for (isim in 1:nsim) {
    theta <- store_theta[isim, ]
    s2 <- as.numeric(crossprod(theta - theta_hat,
        solve(Qtheta, theta - theta_hat)))
    if (s2 < chi2q) {
        beta <- theta[1:(m-2)]
        sig2 <- theta[m-1]
        nu <- theta[m]
        e <- as.numeric(y - X %*% beta)
        llike <- T*(lgamma((nu+1)/2) - lgamma(nu/2)
            - 0.5*log(nu*pi*sig2)) -
            (nu+1)/2*sum(log(1 + e^2/(sig2*nu)))
        logf <- const_f - 0.5*s2  # log f(theta)
        store_w[isim] <- logf -
            (llike + prior(beta, sig2, nu))
    }
}
maxllike <- max(store_w)
log_ml <- log(mean(exp(store_w - maxllike))) + maxllike
log_ml <- -log_ml
cat(sprintf("Log marginal likelihood: %.2f\n", log_ml))
