# Monte Carlo approximation of the posterior probability P(mu < 0 | y) in
# the normal model with known variance.

import numpy as np

np.random.seed(42)

R      = 10000
mu_hat = 0.64
Dmu    = 0.09
mu     = mu_hat + np.sqrt(Dmu)*np.random.randn(R)
g_hat  = np.mean(mu < 0)
print('P(mu < 0 | y) approx:', g_hat)
