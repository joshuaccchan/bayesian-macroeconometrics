# gpr_uncertainty.py
# Empirical-Bayes Gaussian process regression of quarterly US real GDP growth
# on its own lag and lagged macroeconomic uncertainty. The model is
#   y_t = f(x_t) + eps_t,  eps_t ~ N(0, sig2),  f ~ GP(0, k),
#   x_t = (y_{t-1}, U_{t-1})',
# with an ARD squared exponential kernel
#   k(x,x') = sigf2 * exp(-0.5 * sum_j (x_j - x'_j)^2 / lam_j^2).
# The hyperparameters (sigf2, lam_1, lam_2, sig2) are set by maximizing the
# log integrated likelihood (gpr_fit_eb.py); the fitted mean surface is sliced
# at low/median/high lagged growth and plotted against uncertainty. The
# uncertainty index is from Jurado-Ludvigson-Ng (2015); sample 1960Q4-2019Q4.
#
# Requires: gpr_fit_eb.py, gpr_predict.py, shaded_band.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gpr_fit_eb import gpr_fit_eb
from gpr_predict import gpr_predict
from shaded_band import shaded_band

np.random.seed(42)

# load data; keep the pre-COVID sample (through 2019Q4)
tbl = pd.read_csv('gpr_uncertainty_data.csv')
yr = tbl['date'].str.split('Q').str[0].astype(float)
qt = tbl['date'].str.split('Q').str[1].astype(float)
tbl = tbl[(yr + (qt-1)/4) <= 2019.9].reset_index(drop=True)

y = tbl['gdp_growth'].to_numpy()
X = np.column_stack([tbl['gdp_growth_lag1'],
                     tbl['macro_unc_h1_lag1']])  # [lagged growth, lagged unc]
n = X.shape[0]

# center response (zero-mean GP) and standardize each input
my = np.mean(y)
yc = y - my
mx = np.mean(X, axis=0)
sx = np.std(X, axis=0, ddof=1)
Xs = (X - mx)/sx

# empirical-Bayes hyperparameters (maximize the log integrated likelihood)
opts = {'nstart': 12, 'verbose': False}
hp, llf, _ = gpr_fit_eb(yc, Xs, opts)

nm = ['lagged GDP growth', 'lagged uncertainty']
print(f'Empirical-Bayes ARD hyperparameters (n = {n}, '
      f'{tbl["date"].iloc[0]}-{tbl["date"].iloc[-1]})')
print(f'  signal variance sig_f^2 = {hp["sigf2"]:.4f}')
for j in range(2):
    print(f'  length scale lambda_{j+1} = {hp["lam"][j]:6.3f} (std)  '
          f'= {hp["lam"][j]*sx[j]:7.4f} ({nm[j]} units)')
print(f'  noise variance sig^2 = {hp["sig2"]:.4f} (sig = {np.sqrt(hp["sig2"]):.3f})')
print(f'  log integrated likelihood = {llf:.3f}')
less = int(np.argmax(hp['lam']))
print(f'  -> {nm[less]} has the larger length scale (less relevant locally)')

# slices: fix lagged growth, vary uncertainty
# ('hazen' matches MATLAB's quantile method); low / median / high growth
gvals = np.quantile(X[:, 0], [0.10, 0.50, 0.90], method='hazen')
glab = ['10th pct', 'median', '90th pct']
ug = np.linspace(X[:, 1].min(), X[:, 1].max(), 300)  # uncertainty grid

z95 = 1.96
sty = [':', '-', '--']                            # low / median / high
plt.figure(figsize=(9.2, 3.6))

# 95% credible band for the median slice
mum, s2m, _ = gpr_predict(yc, Xs,
                          (np.column_stack([gvals[1]*np.ones_like(ug), ug]) - mx)/sx,
                          hp)
sdm = np.sqrt(s2m)
shaded_band(ug, mum + my - z95*sdm, mum + my + z95*sdm, 0.85)

hm = []
for s in range(3):
    mu, _, _ = gpr_predict(yc, Xs,
                           (np.column_stack([gvals[s]*np.ones_like(ug), ug]) - mx)/sx,
                           hp)
    hm.append(plt.plot(ug, mu + my, color='k', linestyle=sty[s], linewidth=2)[0])

ax = plt.gca()
ax.spines[['top', 'right']].set_visible(False)
plt.xlim(X[:, 1].min(), X[:, 1].max())
plt.xlabel('Macro uncertainty', fontsize=12)
plt.ylabel('Real GDP growth', fontsize=12)
plt.legend(hm,
           [rf'$y_{{t-1}}={gvals[0]:.1f}\%$ (10%-tile)',
            rf'$y_{{t-1}}={gvals[1]:.1f}\%$ (50%-tile)',
            rf'$y_{{t-1}}={gvals[2]:.1f}\%$ (90%-tile)'],
           loc='lower left', fontsize=11, frameon=False)
plt.tight_layout()
plt.savefig('gpr_uncertainty.eps')
plt.show()
