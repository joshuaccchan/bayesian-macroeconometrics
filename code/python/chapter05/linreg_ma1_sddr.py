# linreg_ma1_sddr.py
# Computes the Bayes factor for the ARMA(2,1) model against the AR(2)
# model with independent errors via the Savage-Dickey density ratio.
# The prior for psi is uniform on (-1, 1), so the prior ordinate at
# psi = 0 is 0.5. The posterior ordinate at psi = 0 is estimated by
# averaging the conditional posterior densities p(psi | y, beta, sig2)
# over the posterior draws of (beta, sig2) produced by linreg_ma1.py.

import numpy as np
import matplotlib.pyplot as plt
from loglike_MA1 import loglike_MA1

exec(open('linreg_ma1.py').read())  # estimate the ARMA(2,1) model

n_grid = 400  # number of grid points
psi_grid = np.linspace(-.99, .99, n_grid)  # grid for psi
psi_grid = np.sort(np.append(psi_grid, 0))  # insert 0 explicitly
n_grid = psi_grid.size
idx_0 = np.flatnonzero(psi_grid == 0)[0]  # index for psi = 0
lp_psi = np.zeros(n_grid)  # log posterior density
store_p_psi = np.zeros(n_grid)

for isim in range(nsim):
    beta = store_theta[isim, :3]
    sig2 = store_theta[isim, 3]

    for igrid in range(n_grid):
        psi = psi_grid[igrid]
        lp_psi[igrid] = loglike_MA1(psi,
            y - X @ beta, sig2)
    # normalize to obtain conditional posterior of psi
    p_psi = np.exp(lp_psi - lp_psi.max())
    p_psi = p_psi/np.trapezoid(p_psi, psi_grid)
    store_p_psi = store_p_psi + p_psi
p_psi_hat = store_p_psi/nsim  # posterior of psi
BF_UR = 0.5/p_psi_hat[idx_0]  # BF in favor of MA(1)
print(f'Bayes factor in favor of MA(1): {BF_UR:.3f}')

# plot prior and posterior densities
prior_psi = 0.5*np.ones(n_grid)
plt.figure(figsize=(5, 3.5))
plt.plot(psi_grid, p_psi_hat, 'k-', linewidth=2, label='Posterior')
plt.plot(psi_grid, prior_psi, 'k--', linewidth=1.5, label='Prior')
plt.xlabel(r'$\psi$')
plt.ylabel('Density')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
