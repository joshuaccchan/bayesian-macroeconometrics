# sample_psi_RW.py
# Random-walk Metropolis-Hastings update for the MA(1) parameter psi
# in a linear regression model with MA(1) errors. The invertibility
# restriction |psi| < 1 is enforced by assigning zero acceptance
# probability to proposals outside the admissible region.

import numpy as np
from loglike_MA1 import loglike_MA1


def sample_psi_RW(psi, e, sig2, g_var):
    """RW-MH update for the MA(1) parameter psi.

    Inputs:
      psi  : current value of the MA(1) parameter
      e    : T-vector of regression residuals (y - X*beta)
      sig2 : innovation variance sigma^2
      g_var: proposal variance for the random-walk step

    Outputs:
      psi   : updated value of the MA(1) parameter
      accept: acceptance indicator (True if accepted, False otherwise)
    """
    psi_c = psi + np.sqrt(g_var)*np.random.randn()  # propose candidate

    # check invertibility (0.999 is a numerical buffer)
    if abs(psi_c) < 0.999:
        log_alpha = loglike_MA1(psi_c, e, sig2) \
            - loglike_MA1(psi, e, sig2)
    else:
        log_alpha = -np.inf

    accept = (np.log(np.random.rand()) < log_alpha)  # accept/reject step
    if accept:
        psi = psi_c
    return psi, accept
