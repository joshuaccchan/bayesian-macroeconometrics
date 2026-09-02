# VAR_indep_IR.py
# Structural impulse response analysis of the oil market using a VAR(p) with
# an independent normal and inverse-Wishart prior. A two-block Gibbs
# sampler draws the VAR coefficients and the error covariance; at each
# post-burn-in draw the impulse responses to the three structural shocks
# (recursively identified) are computed with construct_IR.py, and the
# posterior means and pointwise 90% credible bands are plotted.
#
# Requires: construct_IR.py, plotCI.py

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt

from construct_IR import construct_IR
from plotCI import plotCI

np.random.seed(42)   # for reproducibility

p = 24
nsim = 10000
burnin = 1000
n_hz = 19   # impulse response horizon: 0 to 18 months

# load data
data = pd.read_csv('oil_SVAR_data.csv').iloc[:, 1:4].to_numpy()
Y_all = data              # full sample: 1973M2-2019M12
Y0 = Y_all[:p, :]         # initial conditions
Y = Y_all[p:, :]          # estimation sample
T, n = Y.shape
k = 1 + n*p   # number of coefficients per equation

# prior hyperparameters
# prior mean: variable 1 (growth rate) -> 0;
# variables 2,3 (levels) -> random walk
beta0 = np.zeros(n*k)
for j in range(2, n+1):
    beta0[(j-1)*k + j] = 1   # first own lag = 1 for level variables
iVbeta = np.eye(n*k)/100
nu0 = n + 2
S0 = np.eye(n)

# construct the T x k regressor matrix Z
tmpY = np.vstack((Y0[-p:, :], Y))
Z = np.zeros((T, n*p))
for i in range(1, p+1):
    Z[:, (i-1)*n:i*n] = tmpY[p-i:p+T-i, :]
Z = np.column_stack((np.ones(T), Z))
ZZ = Z.T @ Z
ZY = Z.T @ Y

# initialize the Gibbs sampler at OLS
A = np.linalg.lstsq(Z, Y, rcond=None)[0]
beta = A.flatten(order='F')
E = Y - Z @ A
Sig = E.T @ E/T
iSig = np.linalg.solve(Sig, np.eye(n))

# storage for impulse responses: nsim x n_hz x n variables x n shocks
store_yIR = np.zeros((nsim, n_hz, n, n))

for isim in range(nsim + burnin):
    # sample beta
    Kbeta = iVbeta + np.kron(iSig, ZZ)
    Cbeta = np.linalg.cholesky(Kbeta)
    beta_hat = solve_triangular(
        Cbeta.T,
        solve_triangular(Cbeta, iVbeta @ beta0 + (ZY @ iSig).flatten(order='F'),
                         lower=True),
        lower=False)
    beta = beta_hat + solve_triangular(Cbeta.T, np.random.randn(n*k), lower=False)

    # sample Sigma from IW(S0 + E'E, nu0 + T): if Zw is a (nu0+T) x n standard
    # normal matrix and C = chol(S0 + E'E), then C*(Zw'Zw)^{-1}*C' ~ IW
    E = Y - Z @ beta.reshape(k, n, order='F')
    CS = np.linalg.cholesky(S0 + E.T @ E)
    Zw = np.random.randn(nu0 + T, n)
    Sig = CS @ np.linalg.solve(Zw.T @ Zw, CS.T)
    Sig = (Sig + Sig.T)/2
    iSig = np.linalg.solve(Sig, np.eye(n))

    # compute impulse responses
    if isim >= burnin:
        isave = isim - burnin
        for jj in range(n):
            # normalize: each shock raises the oil price
            # shock 1 (supply): negative supply shock -> raise price
            # shocks 2,3 (demand): positive demand shock -> raise price
            shock = np.zeros(n)
            if jj == 0:
                shock[jj] = -1   # negative oil supply shock
            else:
                shock[jj] = 1    # positive demand shock
            yIR = construct_IR(beta, Sig, n_hz, shock)
            store_yIR[isave, :, :, jj] = yIR

# cumulate oil production responses (variable 1 is a growth rate)
store_yIR[:, :, 0, :] = np.cumsum(store_yIR[:, :, 0, :], axis=1)

# posterior mean and 90% credible intervals
yIR_mean = np.mean(store_yIR, axis=0)
yIR_lo = np.quantile(store_yIR, .05, axis=0)
yIR_hi = np.quantile(store_yIR, .95, axis=0)

# print the cumulated impact-plus-18-month responses of the real oil price
print('Response of the real price of oil at horizon 18 (post. mean [90% CI]):')
shocknames = ['Oil supply shock', 'Aggregate demand shock',
              'Oil-specific demand shock']
for jj in range(n):
    print('  %s: %.2f [%.2f, %.2f]'
          % (shocknames[jj], yIR_mean[-1, 2, jj], yIR_lo[-1, 2, jj],
             yIR_hi[-1, 2, jj]))

# plot: 3 x 3 grid (rows = variables, columns = shocks)
varnames = ['Oil production', 'Real activity', 'Real price of oil']
hz = np.arange(n_hz)

plt.figure(figsize=(8, 4))
for ii in range(n):       # response variable (row)
    for jj in range(n):   # shock (column)
        plt.subplot(n, n, ii*n + jj + 1)
        plotCI(hz, yIR_lo[:, ii, jj], yIR_hi[:, ii, jj])
        plt.plot(hz, yIR_mean[:, ii, jj], 'k', linewidth=1.5)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.xlim(-0.5, n_hz-1)
        yl = plt.ylim()
        plt.ylim(min(yl[0], -0.5), max(yl[1], 0.5))
        if ii == 0:
            plt.title(shocknames[jj], fontsize=9)
        if jj == 0:
            plt.ylabel(varnames[ii], fontsize=9)
        if ii == n-1:
            plt.xlabel('Months')
plt.tight_layout()
plt.savefig('oil_SVAR_IR.eps')
plt.show()
