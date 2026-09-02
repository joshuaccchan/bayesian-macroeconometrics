# linreg_t_gd.py
# Computes the log marginal likelihood of the AR(2) model with
# Student-t errors for US PCE inflation via the modified harmonic mean
# (Geweke-Draper) estimator. The auxiliary density f is taken as a
# multivariate normal approximation to the posterior with covariance
# Q_theta, truncated to the (1-alpha)-quantile ellipsoid of the chi^2_m
# distribution to satisfy the thin-tail condition of Geweke (1999).

import numpy as np
from scipy import stats
from scipy.special import gammaln
from lmvnpdf import lmvnpdf
from ligampdf import ligampdf

exec(open('linreg_t.py').read())  # estimate the t model

nsim, m = store_theta.shape
T = y.size

# log prior
prior = lambda b, s, n: lmvnpdf(b, beta0, np.linalg.inv(iVbeta)) \
    + ligampdf(s, nu0, S0) + np.log(1/(nu_ub - 2))

theta_hat = store_theta.mean(axis=0)
Qtheta = np.cov(store_theta, rowvar=False)
alp = .05  # significance level for truncation
chi2q = stats.chi2.ppf(1 - alp, m)

# Cholesky for stable logdet and quadratic forms
L = np.linalg.cholesky(Qtheta)
logdetQ = 2*np.sum(np.log(np.diag(L)))

# log normalizing constant for f
const_f = -0.5*m*np.log(2*np.pi) - 0.5*logdetQ - np.log(1 - alp)

store_w = np.full(nsim, -np.inf)
for isim in range(nsim):
    theta = store_theta[isim, :]
    s2 = (theta - theta_hat) @ np.linalg.solve(Qtheta, theta - theta_hat)
    if s2 < chi2q:
        beta = theta[:m-2]
        sig2 = theta[m-2]
        nu = theta[m-1]
        e = y - X @ beta
        llike = T*(gammaln((nu+1)/2) - gammaln(nu/2)
            - 0.5*np.log(nu*np.pi*sig2)) \
            - (nu+1)/2*np.sum(np.log(1 + e**2/(sig2*nu)))
        logf = const_f - 0.5*s2  # log f(theta)
        store_w[isim] = logf \
            - (llike + prior(beta, sig2, nu))
maxllike = store_w.max()
log_ml = np.log(np.mean(np.exp(store_w - maxllike))) + maxllike
log_ml = -log_ml
print(f'Log marginal likelihood: {log_ml:.2f}')
