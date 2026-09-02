# pred_VAR_SV.R
# Computes the one-step-ahead posterior predictive mean
# and log predictive likelihood of y_{t, var_idx} from the VAR with
# Cholesky stochastic volatility,
#     y_t = x_t' A  + eps_t,    eps_t ~ N(0, Sig_t),
#     Sig_t^{-1} = L' D_t^{-1} L,    D_t = diag(exp(h_{1t}),...,exp(h_{nt})),
#     h_{i,t} = h_{i,t-1} + u_{i,t}^h,    u_{i,t}^h ~ N(0, sigh2_i),
# with the independent Minnesota prior beta ~ N(beta0, diag(V_Minn)),
# N(l0, V_l) prior on the free elements l of L,
# IG(nu_h, S_h) prior on each sigh2_i, and
# N(m_h0, V_h0) prior on h_0 = (h_{1,0},...,h_{n,0})'. Five-block Gibbs
# sampler following Section 13.1.2.
#
# Requires: SURform2.R, SVRW.R
#
# Inputs:
# Yt:       T x n matrix of observations up to time t-1
# Z:        T x k matrix of regressors (intercept + p lags), k = 1+n*p
# xt:       length-k regressor row for forecasting y_t
# beta0:    (nk)-vector Minnesota prior mean of beta
# V_Minn:   (nk)-vector of diagonal entries of the Minnesota prior covariance
# nsim:     scalar; number of post-burnin MCMC draws
# burnin:   scalar; number of burnin draws
# var_idx:  scalar; index of the variable to forecast
# yreal:    scalar; realized value of y_{t, var_idx} (for LPL)
#
# Outputs (returned in a list):
# pf:   posterior predictive mean of y_{t, var_idx}
# lpl:  log of the predictive density at yreal, computed as the
#       log-mean-exp over MCMC draws of mixture-of-Gaussians weights

suppressMessages(library(Matrix))
source("SURform2.R")
source("SVRW.R")

pred_VAR_SV <- function(Yt, Z, xt, beta0, V_Minn, nsim, burnin,
                        var_idx, yreal) {

    T <- nrow(Yt)
    n <- ncol(Yt)
    k <- ncol(Z)
    m <- n*(n - 1)/2
    iVbeta <- Diagonal(x = 1/V_Minn)

    # SV-specific priors
    l0   <- numeric(m);          iVl  <- diag(m)
    m_h0 <- numeric(n);          iVh0 <- diag(n)/10
    nu_h <- 3*rep(1, n);         S_h  <- 0.1*rep(1, n)

    # indices of the strictly lower-triangular elements of L in row-major
    # order, so that L[L_id] = l_vec corresponds to
    #   l_vec = (l_{21}, l_{31}, l_{32}, l_{41}, l_{42}, l_{43}, ...)'
    L_id <- numeric(m)
    ii <- 0
    for (i in 2:n) {
        for (j in 1:(i-1)) {
            ii <- ii + 1
            L_id[ii] <- i + (j-1)*n
        }
    }

    # SUR-form regressors and time-stacked observations
    X <- SURform2(Z, n)
    y <- as.vector(t(Yt))

    # initialize the Gibbs sampler at OLS
    A <- qr.solve(Z, Yt)
    beta <- as.vector(A)
    E <- Yt - Z %*% A
    h0 <- log(diag(crossprod(E)/T))
    h  <- matrix(h0, T, n, byrow = TRUE)
    sigh2 <- 0.1*rep(1, n)
    l_vec <- numeric(m)
    L <- diag(n)

    store_mu  <- numeric(nsim)
    store_var <- numeric(nsim)

    for (isim in 1:(nsim + burnin)) {
        # sample beta
        L[L_id] <- l_vec
        bigL <- kronecker(Diagonal(T), Matrix(L, sparse = TRUE))
        iD <- Diagonal(x = as.vector(t(1/exp(h))))
        iSig <- crossprod(bigL, iD %*% bigL)
        XiSig <- crossprod(X, iSig)
        Kbeta <- as.matrix(iVbeta + XiSig %*% X)
        Cbeta <- chol(Kbeta)   # upper factor: Kbeta = t(Cbeta) %*% Cbeta
        rhs <- as.numeric(iVbeta %*% beta0 + XiSig %*% y)
        beta_hat <- backsolve(Cbeta, backsolve(Cbeta, rhs, transpose = TRUE))
        beta <- as.numeric(beta_hat + backsolve(Cbeta, rnorm(n*k)))

        # sample h equation by equation
        A <- matrix(beta, k, n)
        E <- Yt - Z %*% A
        Eorth <- E %*% t(L)
        ystar <- log(Eorth^2 + 1e-4)
        for (i in 1:n) {
            h[, i] <- SVRW(ystar[, i], h[, i], h0[i], sigh2[i])
        }

        # sample l_vec from the regression eps_t = E_t * l + eta_t
        Em <- matrix(0, T*n, m)
        cE <- 0
        for (ii in 1:(n-1)) {
            Em[seq(ii+1, T*n, by = n), (cE+1):(cE+ii)] <- -E[, 1:ii, drop = FALSE]
            cE <- cE + ii
        }
        iD <- Diagonal(x = as.vector(t(1/exp(h))))
        Kl <- as.matrix(iVl + crossprod(Em, as.matrix(iD %*% Em)))
        Cl <- chol(Kl)
        rhs_l <- as.numeric(iVl %*% l0) +
            as.numeric(crossprod(Em, as.numeric(iD %*% as.vector(t(E)))))
        l_hat <- backsolve(Cl, backsolve(Cl, rhs_l, transpose = TRUE))
        l_vec <- as.numeric(l_hat + backsolve(Cl, rnorm(m)))

        # sample sigh2
        e2 <- (h - rbind(h0, h[1:(T-1), , drop = FALSE]))^2
        # R's rgamma takes shape and scale
        sigh2 <- 1/rgamma(n, shape = nu_h + T/2,
                          scale = 1/(S_h + colSums(e2)/2))

        # sample h0
        Kh0 <- iVh0 + diag(1/sigh2, n, n)
        Ch0 <- chol(Kh0)
        rhs_h0 <- as.numeric(iVh0 %*% m_h0) + h[1, ]/sigh2
        h0_hat <- backsolve(Ch0, backsolve(Ch0, rhs_h0, transpose = TRUE))
        h0 <- as.numeric(h0_hat + backsolve(Ch0, rnorm(n)))

        if (isim > burnin) {
            isave <- isim - burnin
            mu_full <- as.numeric(xt %*% A)
            # forecast h_{T+1} via random walk and the implied Sig_{T+1}
            h_tp1 <- h[T, ] + sqrt(sigh2)*rnorm(n)
            L[L_id] <- l_vec
            invL <- solve(L)
            Sig_tp1 <- invL %*% diag(exp(h_tp1), n, n) %*% t(invL)
            store_mu[isave]  <- mu_full[var_idx]
            store_var[isave] <- Sig_tp1[var_idx, var_idx]
        }
    }

    # aggregate across draws
    pf <- mean(store_mu)
    log_w <- -0.5*log(2*pi*store_var) - 0.5*(yreal - store_mu)^2/store_var
    lpl <- log(mean(exp(log_w - max(log_w)))) + max(log_w)

    list(pf = pf, lpl = lpl)
}
