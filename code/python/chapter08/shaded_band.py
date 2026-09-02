"""shaded_band.py
Plot a grayscale shaded credible band between pointwise lower and
upper bounds. Useful for visualizing posterior credible intervals
for time-varying quantities.

Inputs:
  x    : length-T vector (numeric or datetime)
  lo   : length-T vector, lower bound
  hi   : length-T vector, upper bound
  shade: grayscale level in [0,1] (optional, default = 0.85)

Output:
  h    : handle to the fill object (a PolyCollection)
"""

import numpy as np
import matplotlib.pyplot as plt


def shaded_band(x, lo, hi, shade=0.85):
    # ensure 1-D vectors
    x = np.ravel(x)
    lo = np.ravel(lo)
    hi = np.ravel(hi)

    # plot shaded region
    h = plt.fill_between(x, lo, hi, color=(shade, shade, shade),
                         edgecolor='none')
    return h
