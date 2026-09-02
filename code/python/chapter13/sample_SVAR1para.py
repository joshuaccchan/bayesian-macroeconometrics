"""sample_SVAR1para.py
Jointly samples the persistence phi_i and the
innovation variance sigh2_i of each stationary AR(1) log-volatility
equation
    h_{i,t} = phi_i*h_{i,t-1} + u_{i,t},   u_{i,t} ~ N(0, sigh2_i),
    h_{i,1} ~ N(0, sigh2_i/(1 - phi_i^2)),
given a current draw of h. phi_i is updated via independence-chain
Metropolis-Hastings using a truncated-normal proposal under the prior
phi_i ~ TN_{(-1,1)}(phi_0, V_phi); the acceptance ratio carries the
prefactor sqrt(1 - phi_i^2) from the stationary initial distribution.
sigh2_i is then drawn from its IG full conditional with shape
nu_h(i) + T/2 and scale S_h(i) + ss/2, where
    ss = (1 - phi_i^2)*h_{i,1}^2 + sum_{t>=2}(h_{i,t} - phi_i*h_{i,t-1})^2.

Inputs:
  h:      T x n matrix of log-volatilities
  phi:    length-n current persistence
  sigh2:  length-n current innovation variance
  phi_0:  scalar; prior mean of phi_i
  V_phi:  scalar; prior variance of phi_i
  nu_h:   length-n IG shape parameter
  S_h:    length-n IG scale parameter

Outputs:
  phi:    length-n updated persistence
  sigh2:  length-n updated innovation variance
"""

import numpy as np


def sample_SVAR1para(h, phi, sigh2, phi_0, V_phi, nu_h, S_h):
    T, n = h.shape
    phi = phi.copy()
    sigh2 = sigh2.copy()

    # phi_i via independence-chain Metropolis-Hastings
    for i in range(n):
        hh1 = h[1:, i]
        hh0 = h[:-1, i]
        K_phi = 1/V_phi + (hh0 @ hh0)/sigh2[i]
        phi_hat = (phi_0/V_phi + (hh0 @ hh1)/sigh2[i])/K_phi
        sd_phi = 1/np.sqrt(K_phi)
        phi_prop = phi_hat + sd_phi*np.random.randn()
        attempts = 0
        while abs(phi_prop) >= 1 and attempts < 50:
            phi_prop = phi_hat + sd_phi*np.random.randn()
            attempts += 1
        if abs(phi_prop) >= 1:
            continue
        log_a = 0.5*np.log(1 - phi_prop**2) - 0.5*np.log(1 - phi[i]**2) \
            - h[0, i]**2*(phi[i]**2 - phi_prop**2)/(2*sigh2[i])
        if np.log(np.random.rand()) < log_a:
            phi[i] = phi_prop

    # sigh2_i ~ IG; np.random.gamma uses (shape, scale)
    for i in range(n):
        hh = h[:, i]
        ss = (1 - phi[i]**2)*hh[0]**2 \
            + np.sum((hh[1:] - phi[i]*hh[:-1])**2)
        sigh2[i] = 1/np.random.gamma(nu_h[i] + T/2, 1/(S_h[i] + ss/2))
    return phi, sigh2
