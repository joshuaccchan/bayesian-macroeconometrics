# Gibbs sampler for the normal model with unknown mean mu and variance sig2.

import numpy as np

np.random.seed(42)

# MCMC settings
nsim   = 10000
burnin = 1000

# simulate data from the normal model
T    = 100
mu   = 3
sig2 = 0.1
y    = mu + np.sqrt(sig2)*np.random.randn(T)

# prior hyperparameters (independent normal and inverse-gamma)
mu0   = 0      # prior mean of mu
sig20 = 100    # prior variance of mu
nu0   = 3      # IG shape parameter for sig2
S0    = 0.5    # IG scale parameter for sig2

# initialize the Markov chain
mu   = 0
sig2 = 1

store_theta = np.zeros((nsim, 2))   # columns: [mu, sig2]

for isim in range(nsim + burnin):
    # sample mu
    Dmu    = 1/(1/sig20 + T/sig2)
    mu_hat = Dmu*(mu0/sig20 + np.sum(y)/sig2)
    mu     = mu_hat + np.sqrt(Dmu)*np.random.randn()

    # sample sig2
    # np.random.gamma uses (shape, scale)
    sig2 = 1/np.random.gamma(nu0 + T/2, 1/(S0 + np.sum((y-mu)**2)/2))

    # store draws after burn-in
    if isim >= burnin:
        isave = isim - burnin
        store_theta[isave, :] = [mu, sig2]

# posterior means
theta_mean = store_theta.mean(axis=0)
print('posterior means [mu, sig2]:', theta_mean)
