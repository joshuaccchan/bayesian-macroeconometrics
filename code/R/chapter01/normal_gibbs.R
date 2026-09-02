# Gibbs sampler for the normal model with unknown mean mu and variance sig2.

set.seed(42)

# MCMC settings
nsim   <- 10000
burnin <- 1000

# simulate data from the normal model
T    <- 100
mu   <- 3
sig2 <- 0.1
y    <- mu + sqrt(sig2)*rnorm(T)

# prior hyperparameters (independent normal and inverse-gamma)
mu0   <- 0     # prior mean of mu
sig20 <- 100   # prior variance of mu
nu0   <- 3     # IG shape parameter for sig2
S0    <- 0.5   # IG scale parameter for sig2

# initialize the Markov chain
mu   <- 0
sig2 <- 1

store_theta <- matrix(0, nsim, 2)   # columns: [mu, sig2]

for (isim in 1:(nsim + burnin)) {
    # sample mu
    Dmu    <- 1/(1/sig20 + T/sig2)
    mu_hat <- Dmu*(mu0/sig20 + sum(y)/sig2)
    mu     <- mu_hat + sqrt(Dmu)*rnorm(1)

    # sample sig2
    # R's rgamma takes shape and scale
    sig2 <- 1/rgamma(1, shape = nu0 + T/2, scale = 1/(S0 + sum((y-mu)^2)/2))

    # store draws after burn-in
    if (isim > burnin) {
        isave <- isim - burnin
        store_theta[isave, ] <- c(mu, sig2)
    }
}

# posterior means
theta_mean <- colMeans(store_theta)
cat("posterior means [mu, sig2]:", theta_mean, "\n")
