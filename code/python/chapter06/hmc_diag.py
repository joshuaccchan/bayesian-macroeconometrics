# hmc_diag.py
# Computes MCMC diagnostics for the post burn-in HMC draws produced by
# hmc_demo.py: the inefficiency factor (integrated autocorrelation time),
# the Monte Carlo standard error of the posterior mean, and Geweke's
# convergence Z-statistic, for each coordinate of theta. Requires
# hmc_demo.py, leapfrog.py, inefficiency_factor.py, mcse.py,
# geweke_diag.py, and specvar0.py.

import numpy as np
from inefficiency_factor import inefficiency_factor
from mcse import mcse
from geweke_diag import geweke_diag

exec(open('hmc_demo.py').read())  # obtain post burn-in draws (samples)

L = 50  # truncation lag
IF = inefficiency_factor(samples, L)
MCSE = mcse(samples, L)
Z, pval, _ = geweke_diag(samples)

print('\n            theta_1   theta_2')
print('IF        %8.2f  %8.2f' % (IF[0], IF[1]))
print('MCSE      %8.4f  %8.4f' % (MCSE[0], MCSE[1]))
print('Geweke Z  %8.2f  %8.2f' % (Z[0], Z[1]))
