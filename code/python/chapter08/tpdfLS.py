"""tpdfLS.py
Univariate Student-t density in location-scale form, evaluated at y
with location mu, scale s2, and nu degrees of freedom. The variance is
nu/(nu-2)*s2 for nu > 2.
"""

import numpy as np
from scipy.special import gammaln


def tpdfLS(y, mu, s2, nu):
    s = np.sqrt(s2)
    z = (y - mu)/s
    # the gamma ratio is computed via gammaln: a direct gamma(.)/gamma(.)
    # overflows to inf/inf = nan once nu exceeds about 343, which happens
    # in the DPM of Chapter 8 as soon as a cluster holds ~340 observations
    c = np.exp(gammaln((nu+1)/2) - gammaln(nu/2))/(np.sqrt(nu*np.pi)*s)
    f = c*(1 + (z**2)/nu)**(-(nu+1)/2)
    return f
