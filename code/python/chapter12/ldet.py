# ldet.py
# Computes the log determinant of a symmetric positive definite matrix from
# its lower-triangular Cholesky factor: log|Omega| = 2*sum(log(diag(C))).
#
# Input:
#   Omega : symmetric positive definite matrix
#
# Output:
#   k : log determinant of Omega

import numpy as np


def ldet(Omega):
    return 2*np.sum(np.log(np.diag(np.linalg.cholesky(Omega))))
