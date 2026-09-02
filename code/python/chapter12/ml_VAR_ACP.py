# ml_VAR_ACP.py
# Evaluates the log marginal likelihood of a VAR under the asymmetric
# conjugate prior of Chan (2022), computed equation by equation in the
# recursive structural form (with a zero coefficient prior mean).
#
# Inputs:
#   Y     : T x n matrix of observations
#   Y0    : p0 x n matrix of pre-sample observations (p0 >= p)
#   p     : lag order
#   kappa : [kappa1, kappa2, kappa3] intercept, own-lag and other-lag shrinkage
#   s2    : n-vector of residual variances from univariate AR(p) models
#
# Output:
#   lml : log marginal likelihood

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln


def ml_VAR_ACP(Y, Y0, p, kappa, s2):
    T, n = Y.shape

    # build the lag regressor matrix W (intercept and p lags)
    tmpY = np.vstack((Y0[-p:, :], Y))
    Z = np.zeros((T, n*p))
    for l in range(1, p+1):
        Z[:, (l-1)*n:l*n] = tmpY[p-l:p+T-l, :]
    W = np.column_stack((np.ones(T), Z))

    # accumulate the equation-by-equation contributions
    lml = -T*n/2*np.log(2*np.pi)
    for i in range(1, n+1):
        yi = Y[:, i-1]
        # regressors: contemporaneous variables and lags
        Xi = np.hstack((-Y[:, :i-1], W))

        # prior scale V_i = diag(V_alpha, V_beta), with the
        # Minnesota own- and other-lag shrinkage in V_beta
        vb = np.zeros(1+n*p)
        vb[0] = kappa[0]
        for l in range(1, p+1):
            for r in range(1, n+1):
                idx = (l-1)*n + r
                if r == i:
                    vb[idx] = kappa[1]/(l**2*s2[i-1])   # own lag
                else:
                    vb[idx] = kappa[2]/(l**2*s2[r-1])   # other lag
        Vi = np.concatenate((1/s2[:i-1], vb))
        nu_i = 1 + i/2
        S_i = s2[i-1]/2

        # posterior quantities (the prior mean is zero)
        K = np.diag(1/Vi) + Xi.T @ Xi
        CK = np.linalg.cholesky(K)
        th = solve_triangular(CK.T, solve_triangular(CK, Xi.T @ yi, lower=True),
                              lower=False)
        S_hat = S_i + (yi @ yi - th @ (K @ th))/2

        # add the contribution of equation i
        lml = lml - 0.5*(np.sum(np.log(Vi)) + 2*np.sum(np.log(np.diag(CK)))) \
            + gammaln(nu_i+T/2) + nu_i*np.log(S_i) - gammaln(nu_i) \
            - (nu_i+T/2)*np.log(S_hat)
    return lml
