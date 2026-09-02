# SURform2.R
# Constructs the seemingly unrelated regression (SUR)
# design matrix for an n-equation system with shared regressors.
# Given a T x k matrix Z whose t-th row z_t' contains the regressors
# at time t, it returns the (n*T) x (n*k) sparse matrix whose t-th
# block of n rows is I_n kron z_t', so that
#     y = X * beta,
# where y = vec(Y'), beta = vec(A), and the system Y = Z*A has
# Y (T x n) stacking the n dependent variables in columns and A
# (k x n) the regression coefficients (column j = equation j's k
# coefficients).
#
# Inputs:
# Z:        T x k matrix of regressors (row t is x_t')
# n:        scalar; number of equations in the SUR system
#
# Outputs:
# X:        (n*T) x (n*k) sparse SUR design matrix

suppressMessages(library(Matrix))

SURform2 <- function(Z, n) {
    repZ <- kronecker(Z, matrix(1, n, 1))
    r <- nrow(Z)
    c <- ncol(Z)
    idi <- rep(1:(r*n), each = c)
    idj <- rep(1:(n*c), times = r)
    X <- sparseMatrix(i = idi, j = idj, x = as.vector(t(repZ)),
                      dims = c(n*r, n*c))
    X
}
