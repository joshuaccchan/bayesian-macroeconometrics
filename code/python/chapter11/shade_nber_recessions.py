# shade_nber_recessions.py
# Shades NBER recession periods on the current axes. The x-axis is assumed to
# be in decimal years (e.g. 1947.00, 1947.25, ...). The shading covers the
# postwar recessions through the 2020 COVID recession.
#
# Inputs:
#   ymin, ymax : y-axis limits to cover with the recession shading

import matplotlib.pyplot as plt


def shade_nber_recessions(ymin, ymax):
    # NBER peak-to-trough dates (month precision)
    # [peak_year peak_month  trough_year trough_month]
    R = [
        (1948, 11, 1949, 10),
        (1953,  7, 1954,  5),
        (1957,  8, 1958,  4),
        (1960,  4, 1961,  2),
        (1969, 12, 1970, 11),
        (1973, 11, 1975,  3),
        (1980,  1, 1980,  7),
        (1981,  7, 1982, 11),
        (1990,  7, 1991,  3),
        (2001,  3, 2001, 11),
        (2007, 12, 2009,  6),
        (2020,  2, 2020,  4)]

    def dec(yy, mm):
        return yy + (mm - 1)/12

    ax = plt.gca()
    for (py, pm, ty, tm) in R:
        x1 = dec(py, pm)
        x2 = dec(ty, tm)
        # zorder=0 keeps the shading behind the plotted lines
        ax.fill([x1, x2, x2, x1], [ymin, ymin, ymax, ymax],
                color=(0.9, 0.9, 0.9), edgecolor='none', zorder=0)
