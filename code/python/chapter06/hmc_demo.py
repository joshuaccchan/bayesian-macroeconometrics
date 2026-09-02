# hmc_demo.py
# Demonstrates Hamiltonian Monte Carlo sampling from the bivariate
# target f(theta_1, theta_2) ~
#   exp(-(theta_2 - theta_1^2)^2/20 - (theta_1 - 1)^2/2).
# Generates nsim post burn-in draws using L leapfrog steps with step
# size eps and standard Gaussian momentum.
# Requires leapfrog.py.

import numpy as np
import matplotlib.pyplot as plt
from leapfrog import leapfrog

np.random.seed(42)
# the occasional divergent leapfrog trajectory overflows to inf/nan and
# is rejected by the MH step; silence the warnings (MATLAB is silent)
np.seterr(over='ignore', invalid='ignore')

# log target density and its gradient
logf = lambda q: -0.05*(q[1] - q[0]**2)**2 \
    - 0.5*(q[0] - 1)**2
grad_logf = lambda q: np.array([q[0]/5*(q[1] - q[0]**2)
    - (q[0] - 1), -(q[1] - q[0]**2)/10])

# HMC settings
nsim = 5000
burnin = 1000
eps = 0.8
L = 20

# storage and initialization
theta = np.random.randn(2)
samples = np.zeros((nsim, 2))
accepts = 0

# main HMC loop
for isim in range(nsim + burnin):
    p0 = np.random.randn(2)

    # propose using L-step leapfrog
    thetac, pc = \
        leapfrog(theta, p0, eps, L, grad_logf)

    # Hamiltonians at current and proposed states
    H0 = -logf(theta) + 0.5*(p0 @ p0)
    Hc = -logf(thetac) + 0.5*(pc @ pc)

    # MH accept/reject step
    if np.log(np.random.rand()) < -(Hc - H0):
        theta = thetac
        if isim >= burnin:
            accepts = accepts + 1
    if isim >= burnin:
        samples[isim - burnin, :] = theta

print(f'HMC acceptance rate: {accepts/nsim:.3f}')

# plot: samples and target density contours
t1 = samples[:, 0]
t2 = samples[:, 1]

xg = np.linspace(t1.min(), t1.max(), 300)
yg = np.linspace(t2.min(), t2.max(), 300)
T1, T2 = np.meshgrid(xg, yg)

Z = np.exp(-(0.05*(T2 - T1**2)**2 + 0.5*(T1 - 1)**2))
Z = Z/Z.max()

plt.figure(figsize=(4, 3))
plt.contour(T1, T2, Z, 12, colors='k', linewidths=0.5)
plt.plot(t1, t2, '.k', markersize=2)
plt.xlabel(r'$\theta_1$', fontsize=14)
plt.ylabel(r'$\theta_2$', fontsize=14)
plt.tight_layout()
plt.show()
