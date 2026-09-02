# linreg_t_ce.R
# Computes the log marginal likelihood of the AR(2) model with
# Student-t errors for US PCE inflation via the cross-entropy method.
# The importance density is a product of a multivariate normal density
# for beta and inverse-gamma densities for sig2 and nu, with parameters
# fitted to posterior draws from linreg_t.R. The importance sampling
# estimator is then computed using R_IS independent draws from the
# fitted density. The seed is reset before the importance draws so the
# estimator is reproducible.

source("lmvnpdf.R")
source("ligampdf.R")

source("linreg_t.R")  # estimate the t model

# MATLAB's gamfit (Statistics Toolbox) returns the ML estimates
# [shape, scale] of a gamma distribution. Base R has no gamfit, so the
# ML equations are solved here directly: with
#   s = log(mean(x)) - mean(log(x)),
# the shape a solves log(a) - digamma(a) = s and the scale is mean(x)/a.
gamfit <- function(x) {
    s <- log(mean(x)) - mean(log(x))
    f <- function(a) log(a) - digamma(a) - s
    # f is strictly decreasing from +Inf (a -> 0) to 0 (a -> Inf);
    # bracket the root, then solve
    lo <- 0.1;  hi <- 10
    while (f(lo) < 0) lo <- lo/10
    while (f(hi) > 0) hi <- hi*10
    a <- uniroot(f, c(lo, hi), tol = .Machine$double.eps^0.75)$root
    c(a, mean(x)/a)
}

set.seed(42)   # reset seed for importance sampling
R_IS <- 10000  # number of importance sampling draws (R in the .m)
R_IS <- 20*ceiling(R_IS/20)  # ensure R_IS divisible by 20
m <- ncol(store_theta)
T <- length(y)

# obtain parameters for the IS density
b_hat <- colMeans(store_theta[, 1:(m-2)])
B_hat <- cov(store_theta[, 1:(m-2)]) + 1e-10*diag(m-2)
tmp <- gamfit(1/store_theta[, m-1])
gam1_hat <- tmp[1];  gam2_hat <- 1/tmp[2]
tmp <- gamfit(1/store_theta[, m])
alp1_hat <- tmp[1];  alp2_hat <- 1/tmp[2]

# obtain IS draws from the optimal density
theta_IS <- matrix(0, R_IS, m)
theta_IS[, 1:(m-2)] <- sweep(t(t(chol(B_hat))
    %*% matrix(rnorm((m-2)*R_IS), m-2, R_IS)), 2, b_hat, "+")
theta_IS[, m-1] <- 1/rgamma(R_IS, shape = gam1_hat, scale = 1/gam2_hat)
theta_IS[, m] <- 1/rgamma(R_IS, shape = alp1_hat, scale = 1/alp2_hat)

# construct the prior density
prior <- function(b, s, n) lmvnpdf(b, beta0, solve(iVbeta)) +
    ligampdf(s, nu0, S0) + log(1/(nu_ub - 2)) -
    1e100*((n < 2) || (n > nu_ub))

# construct the IS density
g_IS <- function(b, s, n) lmvnpdf(b, b_hat, B_hat) +
    ligampdf(s, gam1_hat, gam2_hat) +
    ligampdf(n, alp1_hat, alp2_hat)
store_w <- numeric(R_IS)

for (isim in 1:R_IS) {
    theta <- theta_IS[isim, ]
    beta <- theta[1:(m-2)]
    sig2 <- theta[m-1]
    nu <- theta[m]
    e <- as.numeric(y - X %*% beta)
    llike <- T*(lgamma((nu+1)/2) - lgamma(nu/2)
        - 0.5*log(nu*pi*sig2)) -
        (nu+1)/2*sum(log(1 + (e^2)/(sig2*nu)))
    store_w[isim] <- llike + prior(beta, sig2, nu) -
        g_IS(beta, sig2, nu)
}
# point estimate from all R_IS importance weights
maxw_all <- max(store_w)
log_ml <- log(mean(exp(store_w - maxw_all))) + maxw_all

# batch estimates for the numerical standard error
W <- matrix(store_w, R_IS/20, 20)
maxw <- apply(W, 2, max)
bigml <- log(colMeans(exp(sweep(W, 2, maxw, "-")))) + maxw
ml_std <- sd(bigml)/sqrt(20)
cat(sprintf("Log marginal likelihood: %.2f\n", log_ml))
cat(sprintf("Numerical std. error:    %.2f\n", ml_std))
