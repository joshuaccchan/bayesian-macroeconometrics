# trace_plot_demo.py
# Simulates three artificial Markov chains -- a well-mixed AR(1), a
# highly autocorrelated AR(1), and a sticky chain that stays put with
# high probability -- and displays their trace plots side by side.

import numpy as np
import matplotlib.pyplot as plt

R = 2200
burnin = 200
np.random.seed(0)

# -------------------------------------------------------------------------
# (1) Well-mixed AR(1): low persistence
# -------------------------------------------------------------------------
phi1 = 0.20
sig1 = np.sqrt(1 - phi1**2)
x1 = np.zeros(R)
for r in range(1, R):
    x1[r] = phi1*x1[r-1] + sig1*np.random.randn()

# -------------------------------------------------------------------------
# (2) Highly autocorrelated AR(1): high persistence
# -------------------------------------------------------------------------
phi2 = 0.98
sig2 = np.sqrt(1 - phi2**2)
x2 = np.zeros(R)
for r in range(1, R):
    x2[r] = phi2*x2[r-1] + sig2*np.random.randn()

# -------------------------------------------------------------------------
# (3) Sticky chain: stays put with high probability (plateaus)
# -------------------------------------------------------------------------
p_stay = 0.97
sig3 = 0.50
x3 = np.zeros(R)
for r in range(1, R):
    if np.random.rand() < p_stay:
        x3[r] = x3[r-1]
    else:
        x3[r] = x3[r-1] + sig3*np.random.randn()

# Drop burn-in for display
x1 = x1[burnin:]
x2 = x2[burnin:]
x3 = x3[burnin:]

# -------------------------------------------------------------------------
# 1x3 panel of trace plots (black and white)
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 3))

plt.subplot(1, 3, 1)
plt.plot(x1, 'k', linewidth=1)
# plt.title('Well-mixed chain')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel(r'$x^{(r)}$', fontsize=14)

plt.subplot(1, 3, 2)
plt.plot(x2, 'k', linewidth=1)
# plt.title('High autocorrelation')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel(r'$x^{(r)}$', fontsize=14)

plt.subplot(1, 3, 3)
plt.plot(x3, 'k', linewidth=1)
# plt.title('Sticky chain (stuck)')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel(r'$x^{(r)}$', fontsize=14)

plt.tight_layout()
plt.show()
