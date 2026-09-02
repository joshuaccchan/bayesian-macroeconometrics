"""VAR_CSV_o.py
Metropolis-within-Gibbs sampler for a Bayesian VAR with common stochastic
volatility and an outlier component. The VAR coefficients A and the
covariance Sig have a natural conjugate prior.
Requires Minn_NCP.py, sample_CSV_h_ARMH.py, SV_RW_gaussian_approx.py,
shaded_band.py, and inefficiency_factor.py (with specvar0.py).
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cho_solve, solve_triangular
from scipy.stats import invwishart

from Minn_NCP import Minn_NCP
from sample_CSV_h_ARMH import sample_CSV_h_ARMH
from SV_RW_gaussian_approx import SV_RW_gaussian_approx
from shaded_band import shaded_band
from inefficiency_factor import inefficiency_factor

np.random.seed(42)

p = 12   # number of lags (monthly data)
nsims = 30000
burnin = 2000

# load monthly FRED-MD data: the 16 variables of Carriero, Clark, Marcellino
# and Mertens (2024), augmented with nine series spanning interest rates,
# money, credit, and prices for a 25-variable panel
raw = pd.read_csv('FRED-MD.csv')
vars_ = ['RPI', 'DPCERA3M086SBEA', 'INDPRO', 'CUMFNS', 'UNRATE', 'PAYEMS',
         'CES0600000007', 'CES0600000008', 'WPSFD49207', 'PCEPI', 'HOUST',
         'S&P 500', 'EXUSUKx', 'GS5', 'GS10', 'BAAFFM',   # Carriero et al. (2024)
         'FEDFUNDS', 'TB3MS', 'GS1', 'M2SL', 'BUSLOANS', 'CPIAUCSL',
         'OILPRICEx', 'RETAILx', 'PERMIT']                # additional series
data = raw[vars_].to_numpy()
dates = raw.iloc[:, 0].to_numpy()
ok = ~np.isnan(data).any(axis=1)   # keep fully observed months
data = data[ok, :]
dates = dates[ok]
Y0 = data[:p, :]       # pre-sample obs (initial conditions)
Y = data[p:, :]        # estimation sample
dates = dates[p:]
print('FRED-MD: n=%d variables, sample %.2f-%.2f (%d months)'
      % (len(vars_), dates[0], dates[-1], len(dates)))
T, n = Y.shape
k = n*p + 1

# prior hyperparameters
kappa1 = 100     # prior variance on intercepts
kappa2 = 0.2**2  # overall shrinkage on lag coefficients
rw = 0           # zero prior mean
A0, VA, nu0, S0 = Minn_NCP(Y, Y0, p, kappa1, kappa2, rw)
iva = 1/np.diag(VA)   # diagonal of VA^{-1}
phi0 = 0.95
Vphi = 0.1**2                     # common SV: phi ~ N(phi0,Vphi) on (-1,1)
nuh0 = 20
Sh0 = 0.05*(nuh0 - 1)             # sigh2 ~ IG(nuh0,Sh0)
p0a = 10/4
p0b = (1 - 1/48)*120              # outlier prob p_o ~ Beta(p0a,p0b)
ngrid = 100                       # grid points for the U(2,10) component
o_grid = np.concatenate(([1], np.linspace(2, 10, ngrid)))  # support of o_t:
                                  # point mass at 1 plus grid

# construct the regressor matrix Z = [1, y_{t-1}',...,y_{t-p}']
tmpY = np.vstack((Y0[-p:, :], Y))
Z = np.zeros((T, n*p))
for i in range(1, p + 1):
    Z[:, (i-1)*n:i*n] = tmpY[p-i:len(tmpY)-i, :]
Z = np.column_stack((np.ones(T), Z))

# storage
store_A = np.zeros((k, n))
store_Sig = np.zeros((n, n))
store_h = np.zeros((nsims, T))
store_o = np.zeros((nsims, T))
store_theta = np.zeros((nsims, 3))   # [phi, sigh2, po]
counth = 0
countphi = 0

# initialize the Markov chain
phi = phi0
sigh2 = 0.1
o = np.ones(T)
o2 = o**2            # o2_t = o_t^2
po = 1/48
# initialize h with a Gaussian approximation
A_ols = np.linalg.solve(Z.T @ Z + np.diag(iva), Z.T @ Y)
U_ols = Y - Z @ A_ols
C_ols = np.linalg.cholesky(U_ols.T @ U_ols/T)
s2_init = np.sum(solve_triangular(C_ols, U_ols.T, lower=True).T**2, axis=1)
h = SV_RW_gaussian_approx(s2_init, np.mean(np.log(s2_init)), sigh2)
h = h - np.mean(h)   # common SV is normalized to zero mean

print('Starting MCMC for the VAR with common SV and outliers....')
start_time = time.time()
for isim in range(nsims + burnin):
    # sample Sig and A given (h, o2) with Omega^{-1} = diag(exp(-h)./o2)
    iOm = np.exp(-h)/o2
    ZiOm = Z.T*iOm
    KA = np.diag(iva) + ZiOm @ Z
    CKA = np.linalg.cholesky(KA)
    Ahat = cho_solve((CKA, True), iva[:, None]*A0 + ZiOm @ Y)
    Shat = (S0 + A0.T @ (iva[:, None]*A0) + Y.T @ (iOm[:, None]*Y)
            - Ahat.T @ KA @ Ahat)
    Shat = (Shat + Shat.T)/2   # adjust for rounding errors
    Sig = invwishart.rvs(df=nu0 + T, scale=Shat)
    CSig = np.linalg.cholesky(Sig)
    A = Ahat + solve_triangular(CKA.T, np.random.randn(k, n),
                                lower=False) @ CSig.T

    # sample the common log-volatility h via the Laplace-based ARMH step
    U = Y - Z @ A    # reduced-form residuals
    tmp = solve_triangular(CSig, U.T, lower=True).T  # standardized residuals
    s2 = np.sum(tmp**2, axis=1)/o2  # per-period sum of squares over n vars
    h, flag = sample_CSV_h_ARMH(s2, phi, sigh2, h, n, 30)  # kappa=30 for
                                                           # better mixing
    counth += flag

    # sample the outlier states o_t on a grid and the outlier probability p_o
    # (the draw is vectorized over t; the .m file loops over t)
    s2 = np.sum(tmp**2, axis=1)/np.exp(h)  # per-period sum of squares given h
    o_lpri = np.log(np.concatenate(([1 - po], po/ngrid*np.ones(ngrid))))
    llike = -n*np.log(o_grid) - 0.5*s2[:, None]/o_grid**2  # T x (ngrid+1)
    lp = llike + o_lpri
    o_post = np.exp(lp - lp.max(axis=1, keepdims=True))
    o_post = o_post/o_post.sum(axis=1, keepdims=True)
    cdf_o = np.cumsum(o_post, axis=1)
    o = o_grid[(np.random.rand(T)[:, None] > cdf_o).sum(axis=1)]  # inverse CDF
    n_out = np.sum(o > 1)
    po = np.random.beta(p0a + n_out, p0b + T - n_out)
    o2 = o**2

    # sample sigh2
    eh = np.concatenate(([h[0]*np.sqrt(1 - phi**2)], h[1:] - phi*h[:-1]))
    # np.random.gamma uses (shape, scale)
    sigh2 = 1/np.random.gamma(nuh0 + T/2, 1/(Sh0 + np.sum(eh**2)/2))

    # sample phi via an independence-chain MH step
    Kphi = 1/Vphi + np.sum(h[:T-1]**2)/sigh2
    phihat = (phi0/Vphi + h[:T-1] @ h[1:T]/sigh2)/Kphi
    phic = phihat + np.random.randn()/np.sqrt(Kphi)

    def gphi(x):
        return -.5*np.log(sigh2/(1 - x**2)) - .5*(1 - x**2)/sigh2*h[0]**2
    if np.abs(phic) < .9999:
        if np.exp(gphi(phic) - gphi(phi)) > np.random.rand():
            phi = phic
            countphi += 1

    if isim >= burnin:
        isave = isim - burnin
        store_A += A
        store_Sig += Sig
        store_h[isave, :] = h
        store_o[isave, :] = o
        store_theta[isave, :] = [phi, sigh2, po]

    if (isim + 1) % 1000 == 0:
        print('%d loops... ' % (isim + 1))
print('MCMC takes %.1f seconds' % (time.time() - start_time))

# posterior summaries
# the level of h is not separately identified from Sig (only exp(h_t)*Sig is),
# so normalize each draw of h to zero mean before reporting the volatility
store_h = store_h - store_h.mean(axis=1, keepdims=True)
A_mean = store_A/nsims
Sig_mean = store_Sig/nsims
theta_mean = store_theta.mean(axis=0)
h_mean = store_h.mean(axis=0)
vol = np.exp(store_h/2)  # e^{h_t/2}: normalized uncertainty measure
vol_mean = vol.mean(axis=0)
vol_ci = np.quantile(vol, [.05, .95], axis=0).T   # 90% credible interval
o_mean = store_o.mean(axis=0)
o_ci = np.quantile(store_o, [.05, .95], axis=0).T  # 90% credible interval
oprob = (store_o > 1).mean(axis=0)                 # posterior P(o_t > 1)

print('posterior mean p_o: %.3f; months with P(o_t>1)>0.5: %d'
      % (theta_mean[2], np.sum(oprob > 0.5)))
print('acceptance rate (h): %.3f' % (counth/(nsims + burnin)))

# inefficiency factors (integrated autocorrelation times) of the T elements
# of the common log-volatility h, summarizing the mixing of the ARMH step.
# The spectral estimator uses a Bartlett window with truncation lag 500.
IF_h = inefficiency_factor(store_h, 500)   # store_h is nsims x T
print('IF of h: mean %.1f, median %.1f' % (np.mean(IF_h), np.median(IF_h)))

# final results summary
print('phi posterior mean %.3f ; sigh2 posterior mean %.4f'
      % (theta_mean[0], theta_mean[1]))
print('max posterior mean volatility e^{h_t/2}: %.2f (at %.2f)'
      % (vol_mean.max(), dates[vol_mean.argmax()]))

# common volatility e^{h_t/2} with 90% credible interval
plt.figure(figsize=(8, 3))
shaded_band(dates, vol_ci[:, 0], vol_ci[:, 1])
plt.plot(dates, vol_mean, 'k')
# plt.title('common volatility $e^{h_t/2}$')
plt.tight_layout()

# outlier component o_t with 90% credible interval
plt.figure(figsize=(8, 3))
shaded_band(dates, o_ci[:, 0], o_ci[:, 1])
plt.plot(dates, o_mean, 'k')
# plt.title('outlier component $o_t$')
plt.tight_layout()

plt.show()
