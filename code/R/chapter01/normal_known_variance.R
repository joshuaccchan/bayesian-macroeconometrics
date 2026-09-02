# Monte Carlo approximation of the posterior probability P(mu < 0 | y) in
# the normal model with known variance.

set.seed(42)

R      <- 10000
mu_hat <- 0.64
Dmu    <- 0.09
mu     <- mu_hat + sqrt(Dmu)*rnorm(R)
g_hat  <- mean(mu < 0)
cat("P(mu < 0 | y) approx:", g_hat, "\n")
