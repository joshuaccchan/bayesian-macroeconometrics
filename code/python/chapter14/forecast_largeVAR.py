"""forecast_largeVAR.py
Recursive out-of-sample forecasting exercise for the large VAR
chapter: one-step-ahead forecasts of industrial production growth,
CPI inflation, and the unemployment rate from a 25-variable monthly
FRED-MD VAR with p = 12 lags, re-estimated on an expanding window at
each origin from 2000M1 to the end of the sample.

Eight model-prior combinations are compared:
  1) homoskedastic VAR, natural conjugate Minnesota prior
  2) common stochastic volatility, natural conjugate Minnesota prior
  3) Cholesky stochastic volatility, independent Minnesota prior
  4) order-invariant stochastic volatility, independent Minnesota prior
  5) factor stochastic volatility, independent Minnesota prior
  6) order-invariant SV, Minnesota prior with estimated tightness
     (kappa2, kappa3 sampled)
  7) order-invariant SV, horseshoe prior
  8) order-invariant SV, Minnesota-type horseshoe prior (estimated
     tightness plus local horseshoe scales)

The exercise is computationally intensive: with the default settings it
takes several hours (the MATLAB original runs the origins in a parallel
pool; here they run serially).
Requires pred_largeVAR_NCP.py, pred_largeVAR_CSV.py, pred_largeVAR_SV.py,
pred_largeVAR_OISV.py, pred_largeVAR_FSV.py, and their dependencies.
"""

import glob
import os
import time

import numpy as np
import pandas as pd

from pred_largeVAR_NCP import pred_largeVAR_NCP
from pred_largeVAR_CSV import pred_largeVAR_CSV
from pred_largeVAR_SV import pred_largeVAR_SV
from pred_largeVAR_OISV import pred_largeVAR_OISV
from pred_largeVAR_FSV import pred_largeVAR_FSV

p = 12     # number of lags (monthly data)
r = 4      # number of factors in the VAR-FSV
nsim = 5000
burnin = 1000
run_models = [4, 5, 6, 7, 8]   # subset to run, e.g., run_models = [1, 3]
sample_end = 2020   # last date kept (2020.0 = 2019M12). Set to np.inf to
                    # use the full sample

# load monthly FRED-MD data: the 16 variables of Carriero, Clark, Marcellino
# and Mertens (2024), augmented with nine series spanning interest rates,
# money, credit, and prices for a 25-variable panel
raw = pd.read_csv('FRED-MD.csv')
vars_ = ['RPI', 'DPCERA3M086SBEA', 'INDPRO', 'CUMFNS', 'UNRATE', 'PAYEMS',
         'CES0600000007', 'CES0600000008', 'WPSFD49207', 'PCEPI', 'HOUST',
         'S&P 500', 'EXUSUKx', 'GS5', 'GS10', 'BAAFFM',   # Carriero et al. (2024)
         'FEDFUNDS', 'TB3MS', 'GS1', 'M2SL', 'BUSLOANS', 'CPIAUCSL',
         'OILPRICEx', 'RETAILx', 'PERMIT']                # additional series
data = raw[vars_].to_numpy()
# express the log-based series in percentages (100 x log-differences),
# leaving the series already in percentage points unscaled.
in_pp = np.isin(vars_, ['CUMFNS', 'UNRATE', 'CES0600000007', 'GS5',
                        'GS10', 'BAAFFM', 'FEDFUNDS', 'TB3MS', 'GS1'])
data[:, ~in_pp] = 100*data[:, ~in_pp]
dates = raw.iloc[:, 0].to_numpy()
ok = ~np.isnan(data).any(axis=1)   # keep fully observed months
data = data[ok, :]
dates = dates[ok]
pre = dates <= sample_end + 1e-6
data = data[pre, :]
dates = dates[pre]
Y0 = data[:p, :]       # pre-sample obs (initial conditions)
Y = data[p:, :]        # estimation sample
dates = dates[p:]
T, n = Y.shape

# forecast targets: IP, CPI inflation, unemployment rate
targets = np.array([vars_.index('INDPRO'), vars_.index('CPIAUCSL'),
                    vars_.index('UNRATE')])
target_names = ['IP', 'Inflation', 'Unemployment']
q = len(targets)

# prior hyperparameters
kappa1 = 100          # intercept prior variance
kappa2 = 0.2**2       # own-lag tightness (and overall NCP shrinkage)
kappa3 = 0.2**2/4     # cross-lag tightness

# forecast origins: targets from 2000M1 (dates = 2000 + 1/12) to the end
# of the sample
t0 = np.flatnonzero(dates >= 2000 + 1/12 - 1e-6)[0]
origins = np.arange(t0, T)   # forecast target is Y[t, :], estimation uses Y[:t, :]
nfcst = len(origins)
fcst_dates = dates[origins]
actual = Y[np.ix_(origins, targets)]

model_names = ['Homoskedastic', 'Common SV', 'Cholesky SV',
               'Order-invariant SV', 'Factor SV',
               'Minnesota (estimated)', 'Horseshoe', 'Minnesota-type horseshoe']
nmodels = len(model_names)

# res[i, m-1, :] = [pf(1:q), lpl(1:q), lpl_joint] for origin i, model m
res = np.full((nfcst, nmodels, 2*q + 1), np.nan)

# per-origin checkpointing: each completed origin is saved to disk, so an
# interrupted run can resume where it left off, and runs with different
# run_models (e.g., split across machines) are merged automatically. Each
# (origin, model) pair is seeded independently below, so the merged
# results are identical to those from a single full run
ckdir = os.path.join(os.getcwd(), 'fcst_checkpoints')
os.makedirs(ckdir, exist_ok=True)
cktag = os.environ.get('COMPUTERNAME', '') or 'local'
cktag = '%s_m%s' % (cktag, ''.join(str(m) for m in run_models))


def load_checkpoints(ckdir, ifc, nmodels, ncols):
    # merge any saved results for origin ifc, possibly written by different
    # machines or by runs with different run_models subsets
    tmp = np.full((nmodels, ncols), np.nan)
    fls = sorted(glob.glob(os.path.join(ckdir, 'origin_%03d_*.npz' % ifc)))
    for fl in fls:
        S_tmp = np.load(fl)['tmp']
        filled = ~np.isnan(S_tmp)
        tmp[filled] = S_tmp[filled]
    return tmp


def save_checkpoint(ckdir, ifc, tmp, cktag):
    # save the merged results for origin ifc; the machine- and model-specific
    # file name avoids write collisions when the work is split across machines
    np.savez(os.path.join(ckdir, 'origin_%03d_%s.npz' % (ifc, cktag)), tmp=tmp)


def progress_print(incr, total):
    # running progress counter: incr = 0 resets the count and starts the
    # clock; incr = 1 (after each completed origin) prints the count,
    # elapsed time, and an estimate of the time remaining
    if incr == 0:
        progress_print.ndone = 0
        progress_print.t0 = time.time()
        return
    progress_print.ndone += incr
    elapsed = (time.time() - progress_print.t0)/60
    ndone = progress_print.ndone
    print('%d of %d origins done (elapsed %.1f min, est. remaining %.1f min)'
          % (ndone, total, elapsed, elapsed*(total - ndone)/ndone))


print('FRED-MD: n=%d variables, %d forecast origins (%.2f-%.2f)'
      % (n, nfcst, fcst_dates[0], fcst_dates[-1]))
print('models: %s' % '; '.join(model_names[m-1] for m in run_models))

progress_print(0, nfcst)   # reset the counter and start the clock

start_time = time.time()
# the MATLAB original runs this loop as a parfor over a parallel pool;
# joblib.Parallel could parallelize it here in the same way
for ifc in range(nfcst):
    t = origins[ifc]
    Tt = t               # estimation sample: Y[:Tt, :]
    # load any results already saved for this origin and compute only the
    # requested models that are still missing
    tmp = load_checkpoints(ckdir, ifc, nmodels, 2*q + 1)
    todo = [m for m in run_models if np.isnan(tmp[m-1, :]).any()]
    for m in todo:
        np.random.seed(1000000 + 1000*m + t)  # per-(origin, model) seed for
                                              # reproducibility
        # each model is asked to forecast all n variables: the joint
        # LPL is evaluated over the full cross section, while the RMSEs
        # are computed for the three targets only
        if m == 1:
            pf, lj, lm = pred_largeVAR_NCP(Y, Y0, Tt, p,
                                           np.arange(n), kappa1, kappa2, nsim)
        elif m == 2:
            pf, lj, lm = pred_largeVAR_CSV(Y, Y0, Tt, p,
                                           np.arange(n), kappa1, kappa2,
                                           nsim, burnin)
        elif m == 3:
            pf, lj, lm = pred_largeVAR_SV(Y, Y0, Tt, p,
                                          np.arange(n), 'minn', kappa1,
                                          kappa2, kappa3, nsim, burnin)
        elif m == 4:
            pf, lj, lm = pred_largeVAR_OISV(Y, Y0, Tt, p,
                                            np.arange(n), 'minn', kappa1,
                                            kappa2, kappa3, nsim, burnin)[:3]
        elif m == 5:
            pf, lj, lm = pred_largeVAR_FSV(Y, Y0, Tt, p, r,
                                           np.arange(n), 'minn', kappa1,
                                           kappa2, kappa3, nsim, burnin)
        elif m == 6:
            pf, lj, lm = pred_largeVAR_OISV(Y, Y0, Tt, p,
                                            np.arange(n), 'minnH', kappa1,
                                            kappa2, kappa3, nsim, burnin)[:3]
        elif m == 7:
            pf, lj, lm = pred_largeVAR_OISV(Y, Y0, Tt, p,
                                            np.arange(n), 'hs', kappa1,
                                            kappa2, kappa3, nsim, burnin)[:3]
        elif m == 8:
            pf, lj, lm = pred_largeVAR_OISV(Y, Y0, Tt, p,
                                            np.arange(n), 'mahp', kappa1,
                                            kappa2, kappa3, nsim, burnin)[:3]
        tmp[m-1, :] = np.concatenate((pf[targets], lm[targets], [lj]))
    if todo:
        save_checkpoint(ckdir, ifc, tmp, cktag)
    res[ifc, :, :] = tmp
    progress_print(1, nfcst)
print('total time: %.1f minutes' % ((time.time() - start_time)/60))

# evaluation
keep = np.ones(nfcst, dtype=bool)
RMSE = np.full((nmodels, q), np.nan)
sumLPL = np.full(nmodels, np.nan)
for m in run_models:
    err = res[keep, m-1, :q] - actual[keep, :]
    RMSE[m-1, :] = np.sqrt(np.mean(err**2, axis=0))
    sumLPL[m-1] = np.sum(res[keep, m-1, 2*q])

# report the two comparison tables
print('\nForecast performance across SV specifications (Minnesota prior)')
print('%-38s %8s %8s %8s %10s' % ('Specification', *target_names, 'sumLPL'))
for m in [m for m in range(1, 6) if m in run_models]:
    print('%-38s %8.3f %8.3f %8.3f %10.1f'
          % (model_names[m-1], *RMSE[m-1, :], sumLPL[m-1]))
print('\nForecast performance across priors (order-invariant SV)')
print('%-38s %8s %8s %8s %10s' % ('Prior', *target_names, 'sumLPL'))
for m in [m for m in [4, 6, 7, 8] if m in run_models]:
    row_label = 'Minnesota' if m == 4 else model_names[m-1]
    print('%-38s %8.3f %8.3f %8.3f %10.1f'
          % (row_label, *RMSE[m-1, :], sumLPL[m-1]))

np.savez('forecast_largeVAR_results.npz', res=res, actual=actual,
         fcst_dates=fcst_dates, RMSE=RMSE, sumLPL=sumLPL,
         model_names=model_names, target_names=target_names, targets=targets,
         p=p, r=r, nsim=nsim, burnin=burnin, kappa1=kappa1, kappa2=kappa2,
         kappa3=kappa3, run_models=run_models)
