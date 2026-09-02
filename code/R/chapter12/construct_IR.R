# construct_IR.R
# Computes the structural impulse response function of a VAR(p) by iterating
# the VAR forward under two scenarios -- one in which a structural shock hits
# at the impact horizon and one without -- and differencing the two paths.
#
# Inputs:
#   beta  : nk-vector of VAR coefficients, k = 1+n*p
#   Sig   : n x n error covariance matrix
#   n_hz  : number of horizons (including impact, h = 0)
#   shock : n-vector structural shock (e.g., a unit vector e_j)
#
# Output:
#   yIR : n_hz x n impulse responses; row h gives the response at horizon h-1

construct_IR <- function(beta, Sig, n_hz, shock) {
    n <- nrow(Sig)
    p <- (length(beta)/n - 1)/n
    k <- 1 + n*p
    CSig <- t(chol(Sig))   # lower Cholesky factor, Sig = CSig %*% t(CSig)
    # kron(I_n, x_t')*beta = x_t' * A with A = reshape(beta, k, n) column-major
    A <- matrix(beta, k, n)

    # initialize: shocked path starts at CSig*shock, baseline at 0
    tmpZ1 <- matrix(0, p, n)
    tmpZ <- matrix(0, p, n)
    Yt1 <- as.numeric(CSig %*% shock)
    Yt <- numeric(n)
    yIR <- matrix(0, n_hz, n)
    yIR[1, ] <- Yt1

    for (t in 2:n_hz) {
        # update the lagged values for each path
        tmpZ <- rbind(Yt, tmpZ[seq_len(p-1), , drop = FALSE])
        tmpZ1 <- rbind(Yt1, tmpZ1[seq_len(p-1), , drop = FALSE])

        # shocked path: iterate the VAR forward
        Z1 <- as.vector(t(tmpZ1))
        Xt1 <- c(1, Z1)
        Yt1 <- as.numeric(crossprod(Xt1, A))

        # baseline path: iterate the VAR forward
        Z <- as.vector(t(tmpZ))
        Xt <- c(1, Z)
        Yt <- as.numeric(crossprod(Xt, A))

        # impulse response = difference between the two paths
        yIR[t, ] <- Yt1 - Yt
    }
    yIR
}
