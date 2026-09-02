"""gpr_predict.py
Posterior predictive moments for Gaussian process regression with an ARD
squared exponential kernel, at test inputs Xstar given hyperparameters hp:
  E[f(x_{T+1}) | y]   = k_{T+1}' * Ksig^{-1} * y,
  Var[f(x_{T+1}) | y] = k_{T+1,T+1} - k_{T+1}' * Ksig^{-1} * k_{T+1},
where Ksig = K + sig2*I and k_{T+1} is the cross-covariance between a test
input and the T training inputs. Everything goes through the Cholesky factor
of Ksig.

Inputs:
  y     : length-T response
  X     : T-by-d training inputs
  Xstar : m-by-d test inputs
  hp    : dict with keys 'sigf2' (scalar), 'lam' (length-d), 'sig2' (scalar)

Outputs:
  mu  : length-m posterior mean of f at Xstar
  s2f : length-m posterior variance of f at Xstar
  s2y : length-m predictive variance of a noisy observation (= s2f + sig2)
"""

import numpy as np
from scipy.linalg import solve_triangular


def gpr_predict(y, X, Xstar, hp):
    T, d = X.shape
    sigf2 = hp['sigf2']
    lam = np.ravel(hp['lam'])
    sig2 = hp['sig2']

    # ARD scaled squared distances: training-training (M) and training-test (Ms)
    M = np.zeros((T, T))
    Ms = np.zeros((T, Xstar.shape[0]))
    for j in range(d):
        xj = X[:, j]
        M = M + ((xj[:, None] - xj[None, :])**2)/(lam[j]**2)
        Ms = Ms + ((xj[:, None] - Xstar[None, :, j])**2)/(lam[j]**2)
    Ksig = sigf2*np.exp(-0.5*M) + sig2*np.eye(T)  # Ksig = K + sig2 I
    Ks = sigf2*np.exp(-0.5*Ms)                    # cross-covariances k_{T+1}
    C = np.linalg.cholesky(Ksig)                  # lower triangular

    alpha = solve_triangular(C.T, solve_triangular(C, y, lower=True),
                             lower=False)  # Ksig^{-1} y
    mu = Ks.T @ alpha                      # posterior mean k_{T+1}' Ksig^{-1} y

    v = solve_triangular(C, Ks, lower=True)
    s2f = np.maximum(sigf2 - np.sum(v**2, axis=0), 0)  # k_{T+1,T+1} - k' Ksig^{-1} k
    s2y = s2f + sig2                                   # add noise variance for y
    return mu, s2f, s2y
