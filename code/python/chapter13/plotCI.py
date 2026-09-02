"""plotCI.py
Shades a pointwise credible band between the lower curve y1 and the upper
curve y2 over the horizontal grid x, using the fill color C (default light
gray). Used to draw posterior credible intervals.

Inputs:
  x  : horizontal grid (vector)
  y1 : lower band (vector, same length as x)
  y2 : upper band (vector, same length as x)
  C  : optional RGB fill color (default (.85, .85, .85))
"""

import matplotlib.pyplot as plt


def plotCI(x, y1, y2, C=(0.85, 0.85, 0.85)):
    plt.fill_between(x, y1, y2, color=C, edgecolor='none')
