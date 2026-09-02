# VAR_NCP.py
# Forecasting with a VAR(p) under the natural conjugate (normal-inverse-
# Wishart) prior, applied to a four-variable quarterly macro panel. With the
# lag order fixed at p, the script reports the log marginal likelihood and
# runs a recursive (expanding-window) one-step-ahead forecasting exercise for
# PCE inflation from 2000Q1 to 2019Q4, reporting the RMSE and plotting the
# forecasts against the realized series. The computations are closed form, so
# the script is deterministic.
#
# Requires: Minn_NCP.py, estimate_VAR_NCP.py, ml_VAR_NCP.py, mgammaln.py, ldet.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Minn_NCP import Minn_NCP
from estimate_VAR_NCP import estimate_VAR_NCP
from ml_VAR_NCP import ml_VAR_NCP

np.random.seed(42)   # for reproducibility
p = 7
kappa1 = 100
kappa2 = 0.04
rw = 0

# load data
data = pd.read_csv('macro4_Q.csv').iloc[:, 1:].to_numpy()   # drop date column
Y0 = data[:8, :]    # initial conditions: 1960Q1-1961Q4
Y = data[8:240, :]  # 1962Q1-2019Q4
T, n = Y.shape
k = n*p + 1

# construct prior using full sample (for marginal likelihood)
A0, VA, nu0, S0 = Minn_NCP(Y, Y0, p, kappa1, kappa2, rw)
_, KA, _, S_hat = estimate_VAR_NCP(Y, Y0, p, A0, VA, nu0, S0)

print('Log ML (macro4_Q, p=%d): %.2f' % (p, ml_VAR_NCP(VA, S0, nu0, KA, S_hat, T)))

# recursive forecasting from 2000Q1 to 2019Q4
# 2000Q1 corresponds to row (2000-1962)*4+1 = 153
t0 = (2000-1962)*4 + 1
Yfull = np.vstack((Y0, Y))   # stack pre-sample with sample
var_idx = 2                  # PCE inflation (second variable)
nFcst = T - t0 + 1
fcst = np.zeros(nFcst)
actual = Y[t0-1:T, var_idx-1]
fcstDates = 2000 + 0.25*np.arange(nFcst)

for t in range(t0, T+1):
    Yt = Y[:t-1, :]   # estimation sample up to time t-1
    Y0t = Y0
    # construct prior using data up to time t-1
    A0t, VAt, nu0t, S0t = Minn_NCP(Yt, Y0t, p, kappa1, kappa2, rw)
    A_hat = estimate_VAR_NCP(Yt, Y0t, p, A0t, VAt, nu0t, S0t)[0]
    r = Y0.shape[0] + t
    # build x_t = (1, y_{t-1}', ..., y_{t-p}')
    xt = np.concatenate(([1], Yfull[r-p-1:r-1, :][::-1].flatten()))
    yhat = xt @ A_hat
    fcst[t - t0] = yhat[var_idx-1]
err = actual - fcst
RMSE = np.sqrt(np.mean(err**2))

print('1-step PCE inflation forecast (2000Q1-2019Q4): RMSE = %.3f' % RMSE)

plt.figure(figsize=(9, 3.5))
plt.plot(fcstDates, actual, 'k-', linewidth=1.8, label='Actual')
plt.plot(fcstDates, fcst, 'k--', linewidth=1.5, label='Forecast')

plt.xlim(fcstDates[0], fcstDates[-1])
plt.xlabel('Time', fontsize=14)
plt.ylabel('PCE inflation', fontsize=14)
plt.legend(fontsize=12, loc='best')
plt.tight_layout()
plt.savefig('VAR_forecast.eps')
plt.show()
