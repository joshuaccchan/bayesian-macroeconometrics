# SFM.py
# Gibbs sampler for the static factor model fitted to daily exchange-rate
# returns on nine currencies. The model is
#   y_t = A f_t + eps_t,   eps_t ~ N(0, Sigma),   f_t ~ N(0, Omega),
# where A is n x r, lower triangular with ones on the diagonal, and Sigma and
# Omega are diagonal. The 3-block Gibbs sampler draws the factors f, the free
# loadings a, and the variances (Sigma, Omega). After sampling, it estimates
# the log marginal likelihood by the cross-entropy method (SFM_CE.py) and the
# variance decomposition (vardec_SFM.py). Set r to the desired number of factors.
#
# Requires: SFM_CE.py, vardec_SFM.py, logintlike_SFM.py, lmvnpdf.py, ligampdf.py

import time

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

from SFM_CE import SFM_CE
from vardec_SFM import vardec_SFM
from ligampdf import ligampdf

np.random.seed(7)
nsim = 20000
burnin = 1000
R = 10000   # number of importance draws for the cross-entropy method
r = 3       # # of factors
data = pd.read_csv('daily_fx.csv', header=None).to_numpy()
returns = 100*np.log(data[1:, :]/data[:-1, :])
Y = returns

T, n = Y.shape
na = n*r - r*(r+1)//2

# storage
store_a = np.zeros((nsim, na))
store_f = np.zeros((nsim, T*r))
store_sig2 = np.zeros((nsim, n))
store_omega2 = np.zeros((nsim, r))

# prior hyperparameters
a0 = 0
Va = 1   # aij iid N(a0,Va)
nusig2 = 3
Ssig2 = 1*(nusig2-1)*np.ones(n)
nuomega2 = 3
Somega2 = 1*(nuomega2-1)*np.ones(r)


def prior(ax, s, o):
    return (-na/2*np.log(2*np.pi*Va) - 0.5*np.sum((ax-a0)**2/Va)
            + np.sum(ligampdf(s, nusig2, Ssig2))
            + np.sum(ligampdf(o, nuomega2, Somega2)))


# initialize
varY = np.var(Y, axis=0, ddof=1)
sig2 = varY/2                         # diagonal elements of Sigma
omega2 = np.mean(varY)/2*np.ones(r)   # diagonal elements of Omega
a = np.random.randn(na)
A = np.vstack((np.eye(r), np.zeros((n-r, r))))
count_a = 0
for ii in range(1, n):
    nai = min(ii, r)
    A[ii, :nai] = a[count_a:count_a+nai]
    count_a = count_a + nai

tStart = time.time()
for isim in range(nsim + burnin):
    # sample f
    # the precision of f is block diagonal with T identical r x r blocks
    # Kblk = diag(1/omega2) + A'*iSig*A, so all f_t are drawn jointly from
    # the same block Cholesky factor
    AiSig = A.T * (1/sig2)                      # r x n, equals A'*Sigma^{-1}
    Kblk = np.diag(1/omega2) + AiSig @ A
    CKblk = np.linalg.cholesky(Kblk)
    Bf = AiSig @ Y.T                            # r x T; column t is A'*iSig*y_t
    Fhat = solve_triangular(CKblk.T, solve_triangular(CKblk, Bf, lower=True),
                            lower=False)
    F = (Fhat + solve_triangular(CKblk.T, np.random.randn(r, T),
                                 lower=False)).T   # T x r - tth row is f_t
    f = F.flatten()

    # sample a
    count_a = 0
    for i in range(1, n):
        if i < r:   # # of elements in ai
            nai = i
        else:
            nai = r
        Z = F[:, :nai]
        Kai = np.eye(nai)/Va + Z.T @ Z/sig2[i]
        if i < r:
            ai_hat = np.linalg.solve(Kai, a0/Va + Z.T @ (Y[:, i]-F[:, i])/sig2[i])
        else:
            ai_hat = np.linalg.solve(Kai, a0/Va + Z.T @ Y[:, i]/sig2[i])
        ai = ai_hat + solve_triangular(np.linalg.cholesky(Kai).T,
                                       np.random.randn(nai), lower=False)
        A[i, :nai] = ai
        a[count_a:count_a+nai] = ai
        count_a = count_a + nai

    # sample sig2 and omega2
    E = Y - F @ A.T
    sig2 = 1/np.random.gamma(nusig2+T/2, 1/(Ssig2 + np.sum(E**2, axis=0)/2))
    omega2 = 1/np.random.gamma(nuomega2+T/2, 1/(Somega2 + np.sum(F**2, axis=0)/2))

    if (isim+1) % 5000 == 0:
        print('Iteration %d of %d (%.1f%%), elapsed time: %.1f seconds'
              % (isim+1, nsim+burnin, 100*(isim+1)/(nsim+burnin),
                 time.time()-tStart))

    if isim >= burnin:
        i = isim - burnin
        store_a[i, :] = a
        store_f[i, :] = f
        store_sig2[i, :] = sig2
        store_omega2[i, :] = omega2

a_mean = np.mean(store_a, axis=0)
astd = np.std(store_a, axis=0, ddof=1)
f_mean = np.mean(store_f, axis=0).reshape(T, r)
sig2_mean = np.mean(store_sig2, axis=0)
omega2_mean = np.mean(store_omega2, axis=0)

logml_CE, logml_std = SFM_CE(store_a, store_sig2, store_omega2, Y, prior, R)
print('Log marginal likelihood = %.1f (s.e. = %.2f)' % (logml_CE, logml_std))

vd_mean, sys_mean, idio_mean = vardec_SFM(store_a, store_sig2, store_omega2)

varnames = ['AUD', 'CAD', 'EUR', 'JPY', 'CHF', 'GBP', 'KRW', 'NZD', 'TWD']

print('Systematic and idiosyncratic variance shares:')
for i in range(n):
    print('%s: systematic = %.2f, idiosyncratic = %.2f'
          % (varnames[i], sys_mean[i], idio_mean[i]))
