"""sample_SVAR1para_mu.py
This function jointly samples the level mu_i, the persistence phi_i,
and the innovation variance sigh2_i of each stationary AR(1)
log-volatility equation
    h_{i,t} = mu_i + phi_i*(h_{i,t-1} - mu_i) + u_{i,t},
    u_{i,t} ~ N(0, sigh2_i),
    h_{i,1} ~ N(mu_i, sigh2_i/(1 - phi_i^2)),
given a current draw of h. sigh2_i is drawn from its IG full
conditional; phi_i is updated via an independence-chain
Metropolis-Hastings step whose acceptance ratio carries the
sqrt(1 - phi_i^2) prefactor from the stationary initial condition;
mu_i is drawn from its Gaussian full conditional. Setting
free_mu[i] = False fixes mu_i at zero (e.g., for factor
log-volatilities whose scale is absorbed into the loadings).

Inputs:
  h:       T x n matrix of log-volatilities
  mu:      length-n current levels
  phi:     length-n current persistence
  sigh2:   length-n current innovation variances
  mu0:     length-n prior mean of mu_i
  Vmu:     length-n prior variance of mu_i
  phi0:    length-n prior mean of phi_i
  Vphi:    length-n prior variance of phi_i
  nu_h:    length-n IG shape parameter of sigh2_i
  S_h:     length-n IG scale parameter of sigh2_i
  free_mu: length-n boolean; True = sample mu_i, False = fix mu_i at 0

Outputs:
  mu:      length-n updated levels
  phi:     length-n updated persistence
  sigh2:   length-n updated innovation variances
"""

import numpy as np


def sample_SVAR1para_mu(h, mu, phi, sigh2, mu0, Vmu, phi0, Vphi,
                        nu_h, S_h, free_mu):
    T, n = h.shape
    mu = mu.copy()
    phi = phi.copy()
    hd = h - mu   # demeaned log-volatilities

    # sample sigh2_i ~ IG
    e_h = np.vstack((hd[0, :]*np.sqrt(1 - phi**2),
                     hd[1:, :] - phi*hd[:-1, :]))
    # np.random.gamma uses (shape, scale)
    sigh2 = 1/np.random.gamma(nu_h + T/2, 1/(S_h + np.sum(e_h**2, axis=0)/2))

    # sample phi_i via independence-chain Metropolis-Hastings
    Kphi = 1/Vphi + np.sum(hd[:T-1, :]**2, axis=0)/sigh2
    phi_hat = (phi0/Vphi + np.sum(hd[:T-1, :]*hd[1:T, :], axis=0)/sigh2)/Kphi
    phic = phi_hat + 1/np.sqrt(Kphi)*np.random.randn(n)
    for i in range(n):
        def g_phi(x):
            return 0.5*np.log(1 - x**2) - 0.5*(1 - x**2)/sigh2[i]*hd[0, i]**2
        if np.abs(phic[i]) < 0.998:
            if np.exp(g_phi(phic[i]) - g_phi(phi[i])) > np.random.rand():
                phi[i] = phic[i]

    # sample mu_i from its Gaussian full conditional
    for i in range(n):
        if free_mu[i]:
            Kmu = 1/Vmu[i] + ((1 - phi[i]**2) + (T-1)*(1 - phi[i])**2)/sigh2[i]
            mu_hat = (mu0[i]/Vmu[i] + (1 - phi[i]**2)/sigh2[i]*h[0, i]
                      + (1 - phi[i])/sigh2[i]
                      * np.sum(h[1:, i] - phi[i]*h[:-1, i]))/Kmu
            mu[i] = mu_hat + np.random.randn()/np.sqrt(Kmu)
    return mu, phi, sigh2
