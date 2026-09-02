# linreg_t_ce.py
# Computes the log marginal likelihood of the AR(2) model with
# Student-t errors for US PCE inflation via the cross-entropy method.
# The importance density is a product of a multivariate normal density
# for beta and inverse-gamma densities for sig2 and nu, with parameters
# fitted to posterior draws from linreg_t.py. The importance sampling
# estimator is then computed using R independent draws from the fitted
# density. The seed is reset before the importance draws so the
# estimator is reproducible.

import numpy as np
from scipy import stats
from scipy.special import gammaln
from lmvnpdf import lmvnpdf
from ligampdf import ligampdf

exec(open('linreg_t.py').read())  # estimate the t model

np.random.seed(42)  # reset seed for importance sampling
R = 10000  # number of importance sampling draws
R = 20*int(np.ceil(R/20))  # ensure R divisible by 20
m = store_theta.shape[1]
T = y.size

# obtain parameters for the IS density
b_hat = store_theta[:, :m-2].mean(axis=0)
B_hat = np.cov(store_theta[:, :m-2], rowvar=False) + 1e-10*np.eye(m-2)
# ML fit of a gamma to 1/sig2 draws (MATLAB: gamfit returns [shape, scale])
tmp = stats.gamma.fit(1/store_theta[:, m-2], floc=0)
gam1_hat = tmp[0]
gam2_hat = 1/tmp[2]
tmp = stats.gamma.fit(1/store_theta[:, m-1], floc=0)
alp1_hat = tmp[0]
alp2_hat = 1/tmp[2]

# obtain IS draws from the optimal density
theta_IS = np.zeros((R, m))
theta_IS[:, :m-2] = b_hat \
    + (np.linalg.cholesky(B_hat) @ np.random.randn(m-2, R)).T
theta_IS[:, m-2] = 1/np.random.gamma(gam1_hat, 1/gam2_hat, R)
theta_IS[:, m-1] = 1/np.random.gamma(alp1_hat, 1/alp2_hat, R)

# construct the prior density
prior = lambda b, s, n: lmvnpdf(b, beta0, np.linalg.inv(iVbeta)) \
    + ligampdf(s, nu0, S0) + np.log(1/(nu_ub - 2)) \
    - 1e100*((n < 2) or (n > nu_ub))

# construct the IS density
g_IS = lambda b, s, n: lmvnpdf(b, b_hat, B_hat) \
    + ligampdf(s, gam1_hat, gam2_hat) \
    + ligampdf(n, alp1_hat, alp2_hat)
store_w = np.zeros(R)

for isim in range(R):
    theta = theta_IS[isim, :]
    beta = theta[:m-2]
    sig2 = theta[m-2]
    nu = theta[m-1]
    e = y - X @ beta
    llike = T*(gammaln((nu+1)/2) - gammaln(nu/2)
        - 0.5*np.log(nu*np.pi*sig2)) \
        - (nu+1)/2*np.sum(np.log(1 + e**2/(sig2*nu)))
    store_w[isim] = llike + prior(beta, sig2, nu) \
        - g_IS(beta, sig2, nu)
# point estimate from all R importance weights
maxw_all = store_w.max()
log_ml = np.log(np.mean(np.exp(store_w - maxw_all))) + maxw_all

# batch estimates for the numerical standard error
W = store_w.reshape(R//20, 20, order='F')
maxw = W.max(axis=0)
bigml = np.log(np.mean(np.exp(W - maxw), axis=0)) + maxw
ml_std = np.std(bigml, ddof=1)/np.sqrt(20)
print(f'Log marginal likelihood: {log_ml:.2f}')
print(f'Numerical std. error:    {ml_std:.2f}')
