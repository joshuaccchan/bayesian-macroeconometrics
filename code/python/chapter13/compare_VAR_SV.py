"""compare_VAR_SV.py
Recursive one-step-ahead forecasting of PCE inflation across three VAR
specifications: homoskedastic, Cholesky stochastic volatility, and
order-invariant stochastic volatility.

Requires: Minn_indep.py, pred_VAR_homo.py, pred_VAR_SV.py, pred_VAR_OISV.py,
          SURform2.py, SVRW.py, sample_B0.py, SVAR1.py, sample_SVAR1para.py
"""

import time
import numpy as np
import pandas as pd

from Minn_indep import Minn_indep
from pred_VAR_homo import pred_VAR_homo
from pred_VAR_SV import pred_VAR_SV
from pred_VAR_OISV import pred_VAR_OISV

np.random.seed(42)  # for reproducibility
p = 7
nsim = 10000
burnin = 1000
kappa1 = 100          # intercept prior variance
kappa2 = 0.2**2       # own-lag tightness
kappa3 = 0.2**2/4     # cross-lag tightness
rw = 0                # 0 = zero prior mean (growth-rate data); 1 = RW
var_idx = 1           # PCE inflation (second variable, 0-based index)

# load data
data = pd.read_csv('macro4_Q.csv').iloc[:, 1:].to_numpy()  # drop date column
Y0 = data[:8, :]      # initial conditions: 1960Q1-1961Q4
Y = data[8:260, :]    # 1962Q1-2024Q4
T, n = Y.shape
k = 1 + n*p

# recursive forecasting from 2000Q1 to 2024Q4
t0 = (2000 - 1962)*4  # 0-based index of 2000Q1 in Y
Yfull = np.vstack((Y0, Y))
nFcst = T - t0
fcst_homo = np.zeros(nFcst)
LPL_homo = np.zeros(nFcst)
fcst_SV = np.zeros(nFcst)
LPL_SV = np.zeros(nFcst)
fcst_OISV = np.zeros(nFcst)
LPL_OISV = np.zeros(nFcst)
actual = Y[t0:T, var_idx]
fcstDates = 2000 + 0.25*np.arange(nFcst)

print('Recursive one-step-ahead forecasts ...')
start = time.time()
for t in range(t0, T):
    Yt = Y[:t, :]
    Tt = Yt.shape[0]
    yreal = Y[t, var_idx]

    # independent Minnesota prior on beta (data up to t-1)
    beta0, V_Minn = Minn_indep(p, kappa1, kappa2, kappa3, Y0, Yt, rw)[:2]

    # regressor matrix Z shared across models
    tmpY = np.vstack((Y0[-p:, :], Yt))
    Z = np.ones((Tt, k))
    for i in range(1, p + 1):
        Z[:, 1+(i-1)*n:1+i*n] = tmpY[p-i:p+Tt-i, :]

    # regressor row x_t for forecasting Y[t, :]
    r = Y0.shape[0] + t   # 0-based row of Y[t] within Yfull
    xt = np.concatenate(([1.0], Yfull[r-p:r, :][::-1].flatten()))

    # homoskedastic VAR
    fcst_homo[t-t0], LPL_homo[t-t0] = pred_VAR_homo(
        Yt, Z, xt, beta0, V_Minn, nsim, burnin, var_idx, yreal)

    # VAR with Cholesky stochastic volatility
    fcst_SV[t-t0], LPL_SV[t-t0] = pred_VAR_SV(
        Yt, Z, xt, beta0, V_Minn, nsim, burnin, var_idx, yreal)

    # VAR with order-invariant stochastic volatility
    fcst_OISV[t-t0], LPL_OISV[t-t0] = pred_VAR_OISV(
        Yt, Z, xt, beta0, V_Minn, nsim, burnin, var_idx, yreal)

    if (t - t0 + 1) % 10 == 0:
        print(f'  forecast {t - t0 + 1} / {nFcst} done', flush=True)
print(f'Elapsed: {time.time() - start:.1f} sec')

err_homo = actual - fcst_homo
RMSE_homo = np.sqrt(np.mean(err_homo**2))
sumLPL_homo = np.sum(LPL_homo)
print(f'Homoskedastic VAR  : RMSE = {RMSE_homo:.4f}   '
      f'sum LPL = {sumLPL_homo:.4f}')

err_SV = actual - fcst_SV
RMSE_SV = np.sqrt(np.mean(err_SV**2))
sumLPL_SV = np.sum(LPL_SV)
print(f'Cholesky SV VAR    : RMSE = {RMSE_SV:.4f}   '
      f'sum LPL = {sumLPL_SV:.4f}')

err_OISV = actual - fcst_OISV
RMSE_OISV = np.sqrt(np.mean(err_OISV**2))
sumLPL_OISV = np.sum(LPL_OISV)
print(f'Order-invariant SV : RMSE = {RMSE_OISV:.4f}   '
      f'sum LPL = {sumLPL_OISV:.4f}')
