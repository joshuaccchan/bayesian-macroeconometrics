# infmix_logchi2.py
# Dirichlet process mixture (DPM) of normals for the log-chi^2 density,
# estimated with a collapsed Gibbs sampler (clusters are labeled 0-based
# in this Python version).
# Requires: tpdfLS.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from tpdfLS import tpdfLS

nsim = 10000
burnin = 1000

# generate data
np.random.seed(1)
T = 2000
df = 1
y = np.log(np.random.chisquare(df, T))

# true log-chi^2_1 density
ftrue = lambda x: 1/np.sqrt(2*np.pi)*np.exp(0.5*x - 0.5*np.exp(x))

# prior hyperparameters
nu0 = 2
S0 = 1
mu0 = 0
Vmu = 100
a_alpha = 1
b_alpha = 1  # prior on alpha

# grid
ngrid = 500
xgrid = np.linspace(-10, 5, ngrid)
store_mixden = np.zeros((nsim, ngrid))
store_M = np.zeros(nsim)

# machine constants used to avoid log(0), as in the MATLAB version
eps_ = np.finfo(float).eps      # MATLAB eps
realmin = np.finfo(float).tiny  # MATLAB realmin


def local_pred_t(z, Tm, sumy, sumy2, mu0, Vmu, nu0, S0):
    """Posterior predictive density of a normal-inverse-gamma component
    evaluated at z. Integrating out (mu, sig2) under the conjugate base
    measure G_0 yields a Student-t; setting Tm = 0 returns the prior
    predictive f_0 under G_0.

    Inputs:
      z    : scalar or vector of evaluation points (e.g. the grid)
      Tm   : number of observations currently in the cluster
      sumy : sum of the cluster observations, sum_t y_t
      sumy2: sum of squared cluster observations, sum_t y_t^2
      mu0  : prior mean of mu
      Vmu  : prior variance scale, mu | sig2 ~ N(mu0, sig2*Vmu)
      nu0  : prior shape for sig2, sig2 ~ IG(nu0, S0)
      S0   : prior scale for sig2

    Output:
      f    : predictive density evaluated at z (same size as z)
    """
    Kmu = 1/Vmu + Tm
    nu_hat = nu0 + Tm/2
    if Tm == 0:
        mu_hat = mu0
        S_hat = S0
    else:
        mu_hat = (mu0/Vmu + sumy)/Kmu
        S_hat = S0 + 0.5*(sumy2 + mu0**2/Vmu - Kmu*mu_hat**2)
    s2 = (S_hat/nu_hat)*(1 + 1/Kmu)
    nu = 2*nu_hat

    return tpdfLS(z, mu_hat, s2, nu)


# posterior predictive density
pred_t_density = lambda z, Tm, sy, sy2: \
    local_pred_t(z, Tm, sy, sy2, mu0, Vmu, nu0, S0)

# initialize using K-means (kmeans2 with kmeans++ starts, matching MATLAB's
# default kmeans initialization; labels are 0-based)
Minit = 10
_, s = kmeans2(y[:, None], Minit, minit='++')
alpha = np.random.gamma(a_alpha, 1/b_alpha)

# maintain clusters dynamically using sufficient statistics
# for cluster m: Tm[m], sumy[m], sumy2[m]
M = s.max() + 1
Tm = np.zeros(M)
sumy = np.zeros(M)
sumy2 = np.zeros(M)

for m in range(M):
    idx = (s == m)
    Tm[m] = np.sum(idx)
    ym = y[idx]
    sumy[m] = np.sum(ym)
    sumy2[m] = ym @ ym

for isim in range(nsim + burnin):
    # sample alpha
    M = len(Tm)  # current # of clusters
    eta = np.random.beta(alpha + 1, T)
    pi_eta = (a_alpha + M - 1)/(T*(b_alpha - np.log(eta)) + a_alpha + M - 1)
    if np.random.rand() < pi_eta:
        alpha = np.random.gamma(a_alpha + M, 1/(b_alpha - np.log(eta)))
    else:
        alpha = np.random.gamma(a_alpha + M - 1, 1/(b_alpha - np.log(eta)))

    # sample s_t
    for t in range(T):
        M = len(Tm)
        mt = s[t]
        # remove observation t from its current cluster
        Tm[mt] = Tm[mt] - 1
        sumy[mt] = sumy[mt] - y[t]
        sumy2[mt] = sumy2[mt] - y[t]**2

        # if cluster becomes empty, remove it
        if Tm[mt] == 0:
            # move last cluster to mt (if mt not last)
            if mt != M - 1:
                Tm[mt] = Tm[M-1]
                sumy[mt] = sumy[M-1]
                sumy2[mt] = sumy2[M-1]

                # relabel all obs. assigned to the last cluster to mt
                s[s == M-1] = mt
            # shrink arrays
            Tm = Tm[:-1]
            sumy = sumy[:-1]
            sumy2 = sumy2[:-1]
            M = M - 1

        # compute assignment probabilities
        logp = np.zeros(M + 1)
        for m in range(M):
            fm = pred_t_density(y[t], Tm[m], sumy[m], sumy2[m])
            # (eps, realmin) added to avoid log(0)
            logp[m] = np.log(Tm[m] + eps_) + np.log(fm + realmin)

        # prior predictive (new cluster) density f0(y_t)
        f0 = pred_t_density(y[t], 0, 0, 0)
        logp[M] = np.log(alpha + eps_) + np.log(f0 + realmin)

        # normalize safely
        logp = logp - logp.max()
        p = np.exp(logp)
        p = p/np.sum(p)

        # sample new assignment
        newm = int(np.flatnonzero(np.random.rand() <= np.cumsum(p))[0])

        if newm < M:
            # assign to existing cluster
            s[t] = newm
            Tm[newm] = Tm[newm] + 1
            sumy[newm] = sumy[newm] + y[t]
            sumy2[newm] = sumy2[newm] + y[t]**2
        else:
            # create new cluster
            M = M + 1
            s[t] = M - 1
            Tm = np.append(Tm, 1)
            sumy = np.append(sumy, y[t])
            sumy2 = np.append(sumy2, y[t]**2)

    if isim >= burnin:
        isave = isim - burnin
        # posterior predictive
        f0g = pred_t_density(xgrid, 0, 0, 0)
        mixden = (alpha/(alpha + T))*f0g
        for m in range(M):
            fm_g = pred_t_density(xgrid, Tm[m], sumy[m], sumy2[m])
            mixden = mixden + (Tm[m]/(alpha + T))*fm_g
        store_mixden[isave, :] = mixden
        store_M[isave] = M

# posterior summaries
M_mean = np.mean(store_M)
mixden_mean = np.mean(store_mixden, axis=0)
mixden_low = np.percentile(store_mixden, 2.5, axis=0)
mixden_high = np.percentile(store_mixden, 97.5, axis=0)

print(f'posterior mean # of clusters: {M_mean:.2f}')
print(f'max abs deviation of the DPM density estimate from the true '
      f'log-chi^2_1 density: {np.max(np.abs(mixden_mean - ftrue(xgrid))):.4f}')

# plot
fig = plt.figure()
plt.fill_between(xgrid, mixden_low, mixden_high,
                 color=(0.85, 0.85, 0.85), edgecolor='none')

hMean, = plt.plot(xgrid, mixden_mean, 'k-', linewidth=2, label='DPM')
hTrue, = plt.plot(xgrid, ftrue(xgrid), 'k--', linewidth=2,
                  label='true density')

plt.xlim(-10, 5)
plt.xlabel(r'$y$')
plt.ylabel('Density')

ax = plt.gca()
ax.spines[['top', 'right']].set_visible(False)
plt.legend(handles=[hMean, hTrue], loc='best', fontsize=12)

plt.tight_layout()
fig.savefig('infmixture_logchi2.eps')
plt.show()
