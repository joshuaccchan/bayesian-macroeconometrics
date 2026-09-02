# SVAR1.R
# Updates the log-volatility h in the stationary AR(1)
# stochastic volatility model
#     y*_t      = h_t + log(chi^2_1),
#     h_t - mu  = phi*(h_{t-1} - mu) + u_t,   u_t ~ N(0, sigh2),  |phi| < 1,
#     h_1 - mu  ~ N(0, sigh2/(1 - phi^2)),
# using the 7-component Gaussian mixture approximation of Kim, Shephard
# and Chib (1998) and the precision-based sampler of Chan and
# Jeliazkov (2009).
#
# Inputs:
# ystar:  length-T vector of transformed observations, y*_t = log(y_t^2 + c)
# h:      length-T current draw of log-volatility
# mu:     scalar; unconditional mean of h
# phi:    scalar; AR(1) persistence, |phi| < 1
# sigh2:  scalar; innovation variance
#
# Output:
# h:      length-T updated draw of log-volatility

suppressMessages(library(Matrix))

SVAR1 <- function(ystar, h, mu, phi, sigh2) {
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
    q <- q / rowSums(q)
    cdfq <- t(apply(q, 1, cumsum))
    s <- rowSums(tmprand > cdfq) + 1
    d_s <- mj[s]
    iSig_s <- Diagonal(x = 1/sigj2[s])

    # sample h with precision K_h = H'*Sigma_inv*H/sigh2 + iSig_s,
    # where H = I - phi*S1 and Sigma_inv = diag(1-phi^2, 1, ..., 1) encodes
    # the stationary initial variance sigh2/(1-phi^2); the prior is then
    # h ~ N(mu*1_T, sigh2*(H'*Sigma_inv*H)^{-1})
    S1 <- sparseMatrix(i = 2:T, j = 1:(T-1), x = 1, dims = c(T, T))
    H  <- Diagonal(T) - phi*S1
    HiSH <- forceSymmetric(crossprod(H, Diagonal(x = c(1-phi^2, rep(1, T-1))) %*% H))
    Kh <- HiSH/sigh2 + iSig_s
    h_hat <- solve(Kh, mu/sigh2*(HiSH %*% rep(1, T)) + iSig_s %*% (ystar - d_s))
    CKh <- chol(Kh)          # UPPER factor: Kh = t(CKh) %*% CKh
    h <- as.numeric(h_hat + solve(CKh, rnorm(T)))
    h
}
