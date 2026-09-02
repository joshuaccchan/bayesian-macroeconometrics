# linreg_outlier.py
# Gibbs sampler for an AR(2) model of US PCE inflation with a
# discrete outlier component:
#   y_t = beta_1 + beta_2 y_{t-1} + beta_3 y_{t-2} + o_t * eps_t,
#   (eps_t | sig2) ~ N(0, sig2),
# where o_t in {1, 5, 10}, P(o_t=1) = 1-p_o, P(o_t=5) = P(o_t=10) =
# p_o/2. The four blocks are: (beta, sig2) from conjugate updates
# conditional on the latent scales, the discrete indicators {o_t}
# from their posterior over {1, 5, 10}, and the outlier probability
# p_o from its beta full conditional.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shaded_band import shaded_band

np.random.seed(42)
nsim = 50000
burnin = 1000

# load data (column PCECTPI; rows 1960Q1-2024Q4)
data = pd.read_csv('USPCE.csv')['PCECTPI'].to_numpy()[:260]
y0 = data[:4]
y = data[4:]
T = len(y)
xlag1 = np.concatenate(([y0[3]], y[:-1]))
xlag2 = np.concatenate((y0[2:4], y[:-2]))
X = np.column_stack((np.ones(T), xlag1, xlag2))
k = X.shape[1]

# prior hyperparameters
beta0 = np.zeros(k)
iVbeta = np.eye(k)/100
nu0 = 3
S0 = 2
p0a = 10/4
p0b = (1 - 1/16)*40

# initialize the chain
beta = np.linalg.solve(X.T @ X, X.T @ y)
sig2 = np.sum((y - X @ beta)**2)/T
po = 1/16
o = np.ones(T)

store_o = np.zeros((nsim, T))
store_theta = np.zeros((nsim, 5))   # [beta', sig2, po]

o_grid = np.array([1., 5., 10.])    # possible values for o_t
for isim in range(nsim + burnin):
    # sample beta
    io = 1/o**2   # iO is diagonal; keep 1/o^2 as a vector
    Dbeta = np.linalg.solve(iVbeta + (X*io[:, None]).T @ X/sig2, np.eye(k))
    beta_hat = Dbeta @ (iVbeta @ beta0 + X.T @ (io*y)/sig2)
    beta = beta_hat + np.linalg.cholesky(Dbeta) @ np.random.randn(k)

    # sample sig2
    e = (y - X @ beta)/o
    sig2 = 1/np.random.gamma(nu0 + T/2, 1/(S0 + e @ e/2))

    # sample o
    o_lpri = np.log([1-po, po/2, po/2])   # log prior density
    u = (y - X @ beta)/np.sqrt(sig2)
    for tt in range(T):
        lliket = -np.log(o_grid) - .5*u[tt]**2/o_grid**2
        o_post = np.exp(lliket + o_lpri - np.max(lliket))
        o_post = o_post/np.sum(o_post)
        idx = np.argmax(np.random.rand() < np.cumsum(o_post))
        o[tt] = o_grid[idx]

    # sample po
    tmp = np.sum(o > 1)
    po = np.random.beta(p0a + tmp, p0b + T - tmp)

    if (isim + 1) % 5000 == 0:
        print(f'{isim + 1} loops... ')

    # store the parameters
    if isim >= burnin:
        isave = isim - burnin
        store_theta[isave, :] = np.concatenate((beta, [sig2, po]))
        store_o[isave, :] = o

theta_mean = store_theta.mean(axis=0)
theta_CI = np.quantile(store_theta, [.025, .975], axis=0)
o_mean = store_o.mean(axis=0)

print("Posterior mean of [beta', sig2, po]:")
print(theta_mean)
print('95% posterior intervals (rows: 2.5%, 97.5%):')
print(theta_CI)

# Outlier probability: posterior mean and pointwise 95% CI
# (cast to float: np.quantile does not accept boolean arrays)
Iout = (store_o > 1).astype(float)            # nsim-by-T indicator
p_mean = Iout.mean(axis=0)                    # length-T posterior mean

p_lo95 = np.quantile(Iout, 0.025, axis=0)     # pointwise lower
p_hi95 = np.quantile(Iout, 0.975, axis=0)     # pointwise upper

tgrid = np.arange(1, T+1)

plt.figure()

# 95% pointwise credible band
shaded_band(tgrid, p_lo95, p_hi95, 0.85)

# posterior mean
plt.plot(tgrid, p_mean, 'k-', linewidth=1.5, label='Posterior mean')

plt.xlabel(r'$t$', fontsize=14)
plt.ylabel(r'$\mathbb{P}(o_t>1\mid \mathbf{y})$', fontsize=14)
plt.legend(loc='best')
plt.ylim(0, 1)

plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(tgrid, y, 'k-', linewidth=1)
plt.ylabel('Inflation', fontsize=14)

plt.subplot(2, 1, 2)
shaded_band(tgrid, p_lo95, p_hi95, 0.85)
plt.plot(tgrid, p_mean, 'k-', linewidth=1.5)
plt.xlabel(r'$t$', fontsize=14)
plt.ylabel(r'$\mathbb{P}(o_t>1\mid \mathbf{y})$', fontsize=14)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
