# ldet.R
# Computes the log determinant of a symmetric positive definite matrix from
# its Cholesky factor: log|Omega| = 2*sum(log(diag(C))).
# R's chol() returns the UPPER factor, but its diagonal coincides with that of
# the lower factor, so the formula is unchanged.
#
# Input:
#   Omega : symmetric positive definite matrix
#
# Output:
#   k : log determinant of Omega

ldet <- function(Omega) {
    2*sum(log(diag(chol(Omega))))
}
