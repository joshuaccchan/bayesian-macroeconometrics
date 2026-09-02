# VAR_NCP_lag.py
# Selects the lag order of the VAR under the natural conjugate prior by
# computing the log marginal likelihood for p = 1,...,8 on macro4_Q, holding
# the estimation sample fixed across lag lengths. The computations are
# closed form, so the script is deterministic.
#
# Requires: Minn_NCP.py, estimate_VAR_NCP.py, ml_VAR_NCP.py, mgammaln.py, ldet.py

import numpy as np
import pandas as pd

from Minn_NCP import Minn_NCP
from estimate_VAR_NCP import estimate_VAR_NCP
from ml_VAR_NCP import ml_VAR_NCP

kappa1 = 100
kappa2 = 0.04
rw = 0

# load data
data = pd.read_csv('macro4_Q.csv').iloc[:, 1:].to_numpy()   # drop date column
Y0 = data[:8, :]    # initial conditions: 1960Q1-1961Q4
Y = data[8:240, :]  # 1962Q1-2019Q4
T = Y.shape[0]

# log marginal likelihood for each lag length
lml = np.zeros(8)
for p in range(1, 9):
    A0, VA, nu0, S0 = Minn_NCP(Y, Y0, p, kappa1, kappa2, rw)
    _, KA, _, S_hat = estimate_VAR_NCP(Y, Y0, p, A0, VA, nu0, S0)
    lml[p-1] = ml_VAR_NCP(VA, S0, nu0, KA, S_hat, T)
    print('log ML (p = %d): %.1f' % (p, lml[p-1]))
