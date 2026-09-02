# VAR_ACP_kappa.py
# Selects the own-lag (kappa2) and other-lag (kappa3) shrinkage
# hyperparameters of the asymmetric conjugate prior for the four-variable
# VAR(7) on macro4_Q by maximizing the closed-form marginal likelihood over
# a grid, and plots the normalized marginal-likelihood contours over the
# two hyperparameters.
#
# Requires: ml_VAR_ACP.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

from ml_VAR_ACP import ml_VAR_ACP

np.random.seed(42)   # for reproducibility

# load data
data = pd.read_csv('macro4_Q.csv').iloc[:, 1:].to_numpy()   # drop date column
p = 7
Y0 = data[:8, :]   # pre-sample (initial conditions)
Y = data[8:, :]    # estimation sample 1962Q1-2019Q4
T, n = Y.shape
kappa1 = 100       # intercept: weak shrinkage

# residual variances from univariate AR(p) models
s2 = np.zeros(n)
tmpY = np.vstack((Y0[-p:, :], Y))
for i in range(n):
    Xi = np.ones((T, 1))
    for l in range(1, p+1):
        Xi = np.column_stack((Xi, tmpY[p-l:p+T-l, i]))
    e = Y[:, i] - Xi @ np.linalg.lstsq(Xi, Y[:, i], rcond=None)[0]
    s2[i] = e @ e/(T - Xi.shape[1])

# evaluate the log marginal likelihood over a grid of (kappa2, kappa3)
k2g = np.linspace(0.005, 0.60, 90)    # own-lag shrinkage
k3g = np.linspace(0.0005, 0.10, 90)   # other-lag shrinkage
LML = np.zeros((k3g.size, k2g.size))
for a in range(k2g.size):
    for b in range(k3g.size):
        LML[b, a] = ml_VAR_ACP(Y, Y0, p, [kappa1, k2g[a], k3g[b]], s2)
dens = np.exp(LML - LML.max())   # normalized surface (flat prior), max = 1

# locate the maximizer and the best symmetric (kappa2 = kappa3) value
ia = np.argmax(LML.max(axis=0))
ib = np.argmax(LML[:, ia])


def fsym(lk):
    return -ml_VAR_ACP(Y, Y0, p, [kappa1, np.exp(lk), np.exp(lk)], s2)


ks = np.exp(minimize_scalar(fsym, bounds=(np.log(1e-4), np.log(1)),
                            method='bounded').x)
print('asymmetric optimum (own,other) = (%.3f, %.4f)' % (k2g[ia], k3g[ib]))
print('best symmetric = %.3f;   log-ML gain over best symmetric = %.2f'
      % (ks, LML.max() + fsym(np.log(ks))))

# plot the marginal-likelihood contours with the key points marked
plt.figure(figsize=(5.6, 4.2))
plt.contour(k2g, k3g, dens, 12, colors='k', linewidths=0.8)   # density contours
plt.plot([0, 0.1], [0, 0.1], '--', color=(0.5, 0.5, 0.5),
         linewidth=1.3)   # symmetric restriction
plt.plot(0.04, 0.04, 'o', markerfacecolor='w', markeredgecolor='k',
         markersize=9, markeredgewidth=1.2)   # natural conjugate, 0.2^2
plt.plot(k2g[ia], k3g[ib], '*k', markerfacecolor='k',
         markersize=14)   # maximizer
plt.xlabel(r'$\kappa_2$ (own lags)', fontsize=14)
plt.ylabel(r'$\kappa_3$ (other lags)', fontsize=14)
plt.xlim(0, 0.6)
plt.ylim(0, 0.1)
plt.tight_layout()
plt.savefig('VAR_ACP_kappa.eps')
plt.show()
