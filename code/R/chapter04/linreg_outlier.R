# linreg_outlier.R
# Gibbs sampler for an AR(2) model of US PCE inflation with a
# discrete outlier component:
#   y_t = beta_1 + beta_2 y_{t-1} + beta_3 y_{t-2} + o_t * eps_t,
#   (eps_t | sig2) ~ N(0, sig2),
# where o_t in {1, 5, 10}, P(o_t=1) = 1-p_o, P(o_t=5) = P(o_t=10) =
# p_o/2. The four blocks are: (beta, sig2) from conjugate updates
# conditional on the latent scales, the discrete indicators {o_t}
# from their posterior over {1, 5, 10}, and the outlier probability
# p_o from its beta full conditional.

source("shaded_band.R")

set.seed(42)
nsim <- 50000; burnin <- 1000

# load data (column PCECTPI; rows 1960Q1-2024Q4)
data <- read.csv("USPCE.csv")$PCECTPI[1:260]
y0 <- data[1:4]
y  <- data[5:length(data)]
T  <- length(y)
xlag1 <- c(y0[4], y[1:(T-1)])
xlag2 <- c(y0[3:4], y[1:(T-2)])
X <- cbind(rep(1, T), xlag1, xlag2)
k <- ncol(X)

# prior hyperparameters
beta0 <- numeric(k); iVbeta <- diag(k)/100
nu0 <- 3; S0 <- 2
p0a <- 10/4; p0b <- (1-1/16)*40

# initialize the chain
beta <- as.numeric(solve(crossprod(X), crossprod(X, y)))
sig2 <- sum((y - X %*% beta)^2)/T
po <- 1/16
o <- rep(1, T)

store_o <- matrix(0, nsim, T)
store_theta <- matrix(0, nsim, 5)   # [beta', sig2, po]

o_grid <- c(1, 5, 10)   # possible values for o_t
for (isim in 1:(nsim + burnin)) {
    # sample beta
    io <- 1/o^2                      # iO is diagonal; keep 1/o^2 as a vector
    Dbeta <- solve(iVbeta + crossprod(X, io*X)/sig2)
    beta_hat <- Dbeta %*% (iVbeta %*% beta0 + crossprod(X, io*y)/sig2)
    beta <- as.numeric(beta_hat + t(chol(Dbeta)) %*% rnorm(k))

    # sample sig2
    e <- as.numeric(y - X %*% beta)/o
    # R's rgamma takes shape and scale
    sig2 <- 1/rgamma(1, shape = nu0 + T/2, scale = 1/(S0 + sum(e*e)/2))

    # sample o
    o_lpri <- log(c(1-po, po/2, po/2))   # log prior density
    u <- as.numeric(y - X %*% beta)/sqrt(sig2)
    for (tt in 1:T) {
        lliket <- -log(o_grid) - .5*u[tt]^2/o_grid^2
        o_post <- exp(lliket + o_lpri - max(lliket))
        o_post <- o_post/sum(o_post)
        idx <- which(runif(1) < cumsum(o_post))[1]
        o[tt] <- o_grid[idx]
    }

    # sample po
    tmp <- sum(o > 1)
    po <- rbeta(1, p0a + tmp, p0b + T - tmp)

    if (isim %% 5000 == 0) {
        cat(isim, "loops... \n")
    }

    # store the parameters
    if (isim > burnin) {
        isave <- isim - burnin
        store_theta[isave, ] <- c(beta, sig2, po)
        store_o[isave, ] <- o
    }
}
theta_mean <- colMeans(store_theta)
theta_CI <- apply(store_theta, 2, quantile, probs = c(.025, .975), type = 5)
o_mean <- colMeans(store_o)

cat("Posterior mean of [beta', sig2, po]:\n")
print(theta_mean)
cat("95% posterior intervals (rows: 2.5%, 97.5%):\n")
print(theta_CI)

# Outlier probability: posterior mean and pointwise 95% CI
# (cast to numeric: quantile does not accept a logical matrix)
Iout <- (store_o > 1) + 0                # nsim-by-T indicator
p_mean <- colMeans(Iout)                 # length-T posterior mean

p_lo95 <- apply(Iout, 2, quantile, probs = 0.025, type = 5)   # pointwise lower
p_hi95 <- apply(Iout, 2, quantile, probs = 0.975, type = 5)   # pointwise upper

tgrid <- 1:T

plot(tgrid, p_mean, type = "n", ylim = c(0, 1), bty = "n",
     xlab = expression(t), ylab = expression(P(o[t] > 1 * "|" * bold(y))))

# 95% pointwise credible band
shaded_band(tgrid, p_lo95, p_hi95, 0.85)

# posterior mean
lines(tgrid, p_mean, col = "black", lwd = 1.5)

legend("top", legend = "Posterior mean", lty = 1, lwd = 1.5, bty = "n")

op <- par(mfrow = c(2, 1))

plot(tgrid, y, type = "l", col = "black", lwd = 1, bty = "n",
     xlab = "", ylab = "Inflation")

plot(tgrid, p_mean, type = "n", ylim = c(0, 1), bty = "n",
     xlab = expression(t), ylab = expression(P(o[t] > 1 * "|" * bold(y))))
shaded_band(tgrid, p_lo95, p_hi95, 0.85)
lines(tgrid, p_mean, col = "black", lwd = 1.5)

par(op)
