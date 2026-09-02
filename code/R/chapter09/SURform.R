# This function constructs a sparse SUR/stacked design matrix
#
# Input:
# X: T-by-k design matrix, where row t is x_t'
#
# Output:
# Z: T-by-(T*k) sparse matrix diag(x_1', ..., x_T')

suppressMessages(library(Matrix))

SURform <- function(X) {

    if (!is.numeric(X) || !is.matrix(X))
        stop("X must be a numeric 2-D matrix.")

    T <- nrow(X)
    k <- ncol(X)

    # row indices: each t repeated k times
    row_idx <- rep(1:T, each = k)

    # column indices: 1,2,...,T*k (block for each t is consecutive k columns)
    col_idx <- 1:(T*k)

    # values: stack rows of X as (x_1', x_2', ..., x_T')'
    vals <- as.vector(t(X))

    Z <- sparseMatrix(i = row_idx, j = col_idx, x = vals, dims = c(T, T*k))
    Z
}
