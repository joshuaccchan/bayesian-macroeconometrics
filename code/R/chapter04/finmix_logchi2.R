# finmix_logchi2.R
# Gibbs sampler that fits a four-component normal mixture
#   y_t ~ sum_{m=1}^M w_m * N(mu_m, sig2_m)
# to simulated data from the log chi^2_1 distribution. Component
# indicators are sampled via the inverse-transform method, mixture
# weights from their Dirichlet full conditional (dirirnd.R), and
# (mu_m, sig2_m) from their normal-inverse-gamma full conditional.
# The chain is initialized using K-means clustering.

source("dirirnd.R")
source("shaded_band.R")

set.seed(42)

nsim <- 20000; burnin <- 1000
M <- 4   # # of components

# generate data
T  <- 2000
df <- 1
y  <- log(rchisq(T, df))

# true log-chi^2_1 density
ftrue <- function(x) 1/sqrt(2*pi) * exp(0.5*x - 0.5*exp(x))

# prior hyperparameters
nu0 <- 2; S0 <- 1
mu0 <- 0; Vmu <- 100
a0  <- 2*rep(1, M)

# grid
ngrid <- 500
xgrid <- seq(-10, 5, length.out = ngrid)

store_mixden <- matrix(0, nsim, ngrid)
like <- matrix(0, T, M)

# initialize using k-means
# (R's kmeans has no k-means++ initialization; it only seeds the chain)
km   <- kmeans(y, M, nstart = 10)
s    <- km$cluster
mu   <- as.numeric(km$centers)
sig2 <- numeric(M)
alp  <- numeric(M)

for (m in 1:M) {
    idx <- (s == m)
    sig2[m] <- var(y[idx])
    alp[m]  <- sum(idx)/T
}

for (isim in 1:(nsim + burnin)) {
    # sample (mu_m, sig2_m) for each component
    for (m in 1:M) {
        idx <- (s == m)
        Tm  <- sum(idx)
        ym  <- y[idx]

        Kmum    <- 1/Vmu + Tm
        mum_hat <- (mu0/Vmu + sum(ym))/Kmum
        Sm_hat  <- S0 + 0.5*(sum(ym*ym) + mu0^2/Vmu
                             - mum_hat^2*Kmum)

        # R's rgamma takes shape and scale
        sig2[m] <- 1/rgamma(1, shape = nu0 + Tm/2, scale = 1/Sm_hat)
        mu[m]   <- mum_hat + sqrt(sig2[m]/Kmum)*rnorm(1)
    }

    # sample s
    for (m in 1:M) {
        like[, m] <- dnorm(y, mu[m], sqrt(sig2[m]))
    }
    joint_den <- sweep(like, 2, alp, "*")
    prob      <- joint_den / rowSums(joint_den)
    u         <- runif(T)
    cumprob   <- t(apply(prob, 1, cumsum))
    s         <- 1 + rowSums(u > cumprob)

    # sample alp
    ns <- numeric(M)
    for (m in 1:M) {
        ns[m] <- sum(s == m)
    }
    alp <- as.numeric(dirirnd(ns + a0, 1))

    # store mixture density
    if (isim > burnin) {
        isave  <- isim - burnin
        dengrid <- sapply(1:M, function(m) dnorm(xgrid, mu[m], sqrt(sig2[m])))
        mixden  <- as.numeric(dengrid %*% alp)
        store_mixden[isave, ] <- mixden
    }
}
mixden_mean <- colMeans(store_mixden)
mixden_low  <- apply(store_mixden, 2, quantile, probs = 0.025, type = 5)
mixden_high <- apply(store_mixden, 2, quantile, probs = 0.975, type = 5)

cat("Last draw of (mu_m, sig2_m, w_m) (component labels not identified):\n")
print(cbind(mu, sig2, alp))
cat("Max abs deviation of posterior mean mixture density from truth:",
    max(abs(mixden_mean - ftrue(xgrid))), "\n")

plot(xgrid, mixden_mean, type = "n", xlim = c(-10, 5),
     ylim = range(c(mixden_low, mixden_high, ftrue(xgrid))),
     xlab = expression(y), ylab = "Density", bty = "n")

# 95% pointwise posterior credible band
shaded_band(xgrid, mixden_low, mixden_high, 0.85)

# posterior mean (solid black)
lines(xgrid, mixden_mean, col = "black", lty = 1, lwd = 2)

# true density (dashed black)
lines(xgrid, ftrue(xgrid), col = "black", lty = 2, lwd = 2)

legend("topleft", legend = c("finite mixture", "true density"),
       lty = c(1, 2), lwd = 2, bty = "n")
