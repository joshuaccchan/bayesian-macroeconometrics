# shaded_band.py
# Plot a grayscale shaded credible band between pointwise lower and
# upper bounds. Useful for visualizing posterior credible intervals
# for time-varying quantities.

import numpy as np
import matplotlib.pyplot as plt


def shaded_band(x, lo, hi, shade=0.85):
    """Plot a grayscale shaded band between lo and hi against x.

    Inputs:
      x    : T-vector (numeric or datetime)
      lo   : T-vector, lower bound
      hi   : T-vector, upper bound
      shade: grayscale level in [0,1] (optional, default = 0.85)

    Output:
      h    : handle to the fill object
    """
    h = plt.fill_between(x, lo, hi, color=shade*np.ones(3),
                         edgecolor='none')
    return h
