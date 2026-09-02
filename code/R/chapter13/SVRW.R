# SVRW.R
# One MCMC update of the log-volatility vector h in the random-walk stochastic
# volatility model
#   y*_t = h_t + log(chi^2_1),
#   h_t  = h_{t-1} + u_t,   u_t ~ N(0, sigh2),   h_1 = h0 + u_1,
# using the 7-component Gaussian mixture approximation of Kim, Shephard and
# Chib (1998) and the precision-based sampler of Chan and Jeliazkov (2009). The
# mixture indicators are drawn first, then h is drawn from its Gaussian full
# conditional with a banded precision matrix. The initial state h0 is treated
# as known.
#
# Inputs:
#   ystar : length-T vector of transformed observations, y*_t = log(y_t^2 + c)
#   h     : length-T current draw of the log-volatility
#   h0    : initial log-volatility (state at time 0), scalar
#   sigh2 : innovation variance of the random-walk state equation, scalar
#
# Output:
#   h     : length-T updated draw of the log-volatility

suppressMessages(library(Matrix))

SVRW <- function(ystar, h, h0, sigh2) {

    T <- length(h)

    # 7-component normal mixture approximation for log(chi^2_1)
    pj    <- c(0.0073, .10556, .00002, .04395, .34001, .24566, .2575)
    mj    <- c(-10.12999, -3.97281, -8.56686, 2.77786, .61942,
               1.79518, -1.08819) - 1.2704
    sigj2 <- c(5.79596, 2.61369, 5.17950, .16735, .64009,
               .34023, 1.26261)
    sigj  <- sqrt(sigj2)

    # sample mixture indicators s_t in {1,...,7}
    tmprand <- runif(T)
    q <- matrix(pj, T, 7, byrow = TRUE) *
        dnorm(matrix(ystar, T, 7),
              matrix(h, T, 7) + matrix(mj, T, 7, byrow = TRUE),
              matrix(sigj, T, 7, byrow = TRUE))
    q <- q/rowSums(q)
    cdfq <- t(apply(q, 1, cumsum))
    s <- rowSums(tmprand > cdfq) + 1
    d_s <- mj[s]
    iSig_s <- Diagonal(x = 1/sigj2[s])

    # sample h with precision K_h = H'H/sigh2 + iSig_s, where (Hh)_1 = h_1
    # and (Hh)_t = h_t - h_{t-1}; the term H'H*ones(T,1) = e_1 contributes
    # h0/sigh2 to the first entry of the linear term
    S1 <- sparseMatrix(i = 2:T, j = 1:(T-1), x = 1, dims = c(T, T))
    H  <- Diagonal(T) - S1
    HH <- crossprod(H)
    Kh <- HH/sigh2 + iSig_s
    h_hat <- solve(Kh, h0/sigh2*as.numeric(HH %*% rep(1, T)) +
                       as.numeric(iSig_s %*% (ystar - d_s)))
    CKh <- chol(Kh)   # upper factor: Kh = t(CKh) %*% CKh
    h <- as.numeric(h_hat + solve(CKh, rnorm(T)))
    h
}
