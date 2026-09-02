"""gp_prior_paths.py
Visualize Gaussian process priors through sample paths.

Draws a handful of sample paths from a zero-mean Gaussian process under
two covariance functions and plots them side by side in a 1-by-2 panel:

  (left)  Brownian motion kernel      k(x,x') = min(x,x')
          -> continuous but rough (nowhere differentiable) paths
  (right) squared exponential kernel  k(x,x') = sf2*exp(-(x-x')^2/(2*l^2))
          -> smooth paths

The kernel is the object that encodes prior beliefs about regularity and
persistence: the squared exponential produces smooth curves, while the
Brownian motion kernel produces jagged ones. This reproduces the
"Visualizing a Gaussian Process" illustration in the Gaussian process
regression section. Paths are drawn in grayscale so the figure reads in
black and white.
"""

import numpy as np
import matplotlib.pyplot as plt


def gp_sample(K, npaths):
    # Draw npaths sample paths from N(0,K) via the Cholesky factor, adding a
    # small jitter to the diagonal (increased adaptively) for numerical
    # stability.
    n = K.shape[0]
    jitter = 1e-12
    while True:
        try:
            L = np.linalg.cholesky(K + jitter*np.eye(n))
            break
        except np.linalg.LinAlgError:
            jitter = jitter*10  # not yet positive definite: add more
    F = L @ np.random.randn(n, npaths)
    return F


np.random.seed(42)                     # fix the seed for reproducible paths

x = np.linspace(0, 1, 200)             # input grid on [0,1]
npaths = 5                             # number of sample paths per panel

# --- covariance (Gram) matrices on the grid ---
K_bm = np.minimum(x[:, None], x[None, :])          # Brownian motion kernel
sf2 = 1                                            # signal variance of the SE kernel
lam = 0.15                                         # length scale of the SE kernel
K_se = sf2*np.exp(-(x[:, None] - x[None, :])**2/(2*lam**2))  # squared exponential

# --- draw sample paths  f ~ N(0,K)  (each column is one path) ---
F_bm = gp_sample(K_bm, npaths)
F_se = gp_sample(K_se, npaths)

# --- plot the 1-by-2 panel (black-and-white, grayscale paths) ---
fs = 14                                            # font size
lw = 1                                             # line width
gray = np.tile(np.linspace(0, 0.7, npaths)[:, None], (1, 3))  # black -> light gray
yl = 1.05*np.max(np.abs(np.concatenate([F_bm.ravel(), F_se.ravel()])))

fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

ax = axes[0]
for j in range(npaths):
    ax.plot(x, F_bm[:, j], '-', color=tuple(gray[j]), linewidth=lw)
ax.set_xlim(0, 1.1)
ax.set_ylim(-yl, yl)
ax.set_xlabel(r'$x$', fontsize=fs)
ax.set_ylabel(r'$f(x)$', fontsize=fs)
ax.set_title(r'$k_{\mathrm{BM}}$', fontsize=fs)
ax.spines[['top', 'right']].set_visible(False)

ax = axes[1]
for j in range(npaths):
    ax.plot(x, F_se[:, j], '-', color=tuple(gray[j]), linewidth=lw)
ax.set_xlim(0, 1.1)
ax.set_ylim(-yl, yl)
ax.set_xlabel(r'$x$', fontsize=fs)
ax.set_ylabel(r'$f(x)$', fontsize=fs)
ax.set_title(r'$k_{\mathrm{SE}}$', fontsize=fs)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
fig.savefig('gp_prior_paths.eps')
plt.show()
