# TVPVAR_Primiceri.R
# Application of the TVP-VAR with stochastic volatility to U.S. GDP
# deflator inflation, unemployment, and the 3-month T-bill rate, replicating
# the dataset and sample of Primiceri (2005).
#
# Requires: estimate_TVPVAR.R, construct_IR.R, plotCI.R, SURform.R, SVRW.R
#
# This script is the only producer of Primiceri_results.rds, which stores the
# plot data (the R counterpart of the MATLAB Primiceri_results.mat) so that
# the two figures can be rebuilt without rerunning the sampler.

suppressMessages(library(Matrix))
source("estimate_TVPVAR.R")
source("construct_IR.R")
source("plotCI.R")

set.seed(42)

p      <- 2
nsim   <- 20000
burnin <- 5000
n_hz   <- 21  # impulse response horizon: 0 to 20 quarters

# load data: 1953Q1-2001Q3, matching Primiceri's sample.
# macro_Primiceri_Q.csv is built by build_macro_Primiceri.m from FRED
# series GDPDEF, UNRATE, TB3MS; columns B:D are INFL, UNRATE, TB3MS.
data  <- as.matrix(read.csv("macro_Primiceri_Q.csv")[, 2:4])
Y_all <- data[1:195, ]                   # 1953Q1-2001Q3
Tall <- nrow(Y_all)
n    <- ncol(Y_all)
k  <- 1 + n*p
nk <- n*k
m  <- n*(n-1)/2

# split: first 40 obs for prior calibration, rest for inference
T_train <- 40
Y_train <- Y_all[1:T_train, ]                 # 1953Q1-1962Q4
Y_est   <- Y_all[(T_train+1-p):Tall, ]        # 1962Q3-2001Q3 (carries p lags)
T_eff   <- Tall - T_train                     # 155 obs (1963Q1-2001Q3)
cat(sprintf("Sample: %d obs total, %d for training, T_eff = %d.\n",
            Tall, T_train, T_eff))

# linear indices of free entries below diagonal of L, eq-by-eq order
L_id <- matrix(1:(n^2), n, n)
L_id[!lower.tri(L_id)] <- 0
L_id <- t(L_id)
L_id <- L_id[L_id != 0]

# training-sample OLS for prior centers and scales
T_tr_eff   <- T_train - p
X_tilde_tr <- matrix(0, T_tr_eff, n*p)
for (i in 1:p) {
    X_tilde_tr[, ((i-1)*n+1):(i*n)] <- Y_train[(p-i+1):(T_train-i), ]
}
Z_tr     <- cbind(rep(1, T_tr_eff), X_tilde_tr)
A_tr     <- qr.solve(Z_tr, Y_train[(p+1):T_train, ])
E_tr     <- Y_train[(p+1):T_train, ] - Z_tr %*% A_tr
Sig_tr   <- crossprod(E_tr)/T_tr_eff
beta_ols <- as.vector(t(A_tr))
Vb_ols   <- kronecker(Sig_tr, solve(crossprod(Z_tr)))

# l_OLS from modified-Cholesky Sig_tr = L^{-1} D (L')^{-1}
Lc_tr <- t(chol(Sig_tr))                 # lower Cholesky factor
L_ols <- diag(diag(Lc_tr)) %*% solve(Lc_tr)
L_ols[upper.tri(L_ols)] <- 0
l_ols <- L_ols[L_id]

# per-equation sampling-variance blocks for l_OLS
sig2_full <- diag(Lc_tr)^2
Vl_blocks <- vector("list", n-1)
for (ii in 1:(n-1)) {
    Ein            <- -E_tr[, 1:ii, drop = FALSE]
    Vl_blocks[[ii]] <- sig2_full[ii+1] * solve(crossprod(Ein))
}
Vl_full <- matrix(0, m, m); off <- 0
for (ii in 1:(n-1)) {
    Vl_full[(off+1):(off+ii), (off+1):(off+ii)] <- Vl_blocks[[ii]]
    off <- off + ii
}

# Prior calibration:
kQ <- 0.01; kS <- 0.1
prior <- list()
prior$beta0_mean <- beta_ols
prior$beta0_var  <- 100 * diag(nk)      # diffuse prior on beta_0
prior$l0_mean    <- l_ols
prior$l0_var     <- 100 * diag(m)       # diffuse prior on l_0
prior$h0_mean    <- log(diag(Lc_tr)^2)
prior$h0_var     <- 100 * diag(n)       # diffuse prior on h_0
prior$Q_nu       <- T_train
prior$Q_S0       <- kQ^2 * T_train * Vb_ols
prior$S_nu       <- (1:(n-1)) + 1
prior$S_S0       <- vector("list", n-1)
for (ii in 1:(n-1)) {
    prior$S_S0[[ii]] <- kS^2 * (ii+1) * Vl_blocks[[ii]]
}
prior$sigh2_nu <- 1 * rep(1, n)
prior$sigh2_S0 <- 0.5 * rep(1, n)

# run the Gibbs sampler
cat(sprintf("Running Gibbs sampler for TVP-VAR (nsim=%d, burnin=%d) ...\n",
            nsim, burnin))
t_start <- Sys.time()
out <- estimate_TVPVAR(Y_est, p, prior, nsim, burnin)
cat(sprintf("Elapsed time is %.1f seconds.\n",
            as.numeric(difftime(Sys.time(), t_start, units = "secs"))))
store_beta <- out$store_beta
store_l    <- out$store_l
store_h    <- out$store_h
store_Q    <- out$store_Q
Q_mean <- matrix(colMeans(store_Q), nk, nk)
cat(sprintf("\nQ posterior mean diag stats: min=%.3e median=%.3e max=%.3e\n",
            min(diag(Q_mean)), median(diag(Q_mean)), max(diag(Q_mean))))

dates_q <- 1963 + 0.25*(0:(T_eff-1))

# residual standard deviations: sqrt(diag(Sigma_t)) at each draw
l_3d <- array(store_l, dim = c(nsim, m, T_eff))
l21  <- l_3d[, 1, ]
l31  <- l_3d[, 2, ]
l32  <- l_3d[, 3, ]
d1   <- exp(store_h[, , 1])
d2   <- exp(store_h[, , 2])
d3   <- exp(store_h[, , 3])

sig2_red <- array(0, dim = c(nsim, T_eff, n))
sig2_red[, , 1] <- d1
sig2_red[, , 2] <- l21^2 * d1 + d2
sig2_red[, , 3] <- (l32*l21 - l31)^2 * d1 + l32^2 * d2 + d3
sig_red <- sqrt(sig2_red)

# quantile type = 5 (Hazen) reproduces MATLAB's quantile
sig_med <- apply(sig_red, c(2, 3), quantile, probs = 0.50, type = 5)
sig_lo  <- apply(sig_red, c(2, 3), quantile, probs = 0.16, type = 5)
sig_hi  <- apply(sig_red, c(2, 3), quantile, probs = 0.84, type = 5)

# Figure 1: residual standard deviations over time
resnames <- c("Inflation", "Unemployment", "3-month T-bill")
setEPS()
postscript("Primiceri_sigma.eps", width = 7, height = 5)
par(mfrow = c(n, 1), mar = c(3, 4, 1, 1))
for (ii in 1:n) {
    plot(dates_q, sig_med[, ii], type = "n",
         xlim = c(dates_q[1], dates_q[T_eff]),
         ylim = range(sig_lo[, ii], sig_hi[, ii]),
         xlab = "", ylab = "", bty = "n")
    plotCI(dates_q, sig_lo[, ii], sig_hi[, ii])
    lines(dates_q, sig_med[, ii], col = "black", lwd = 1.5)
    if (ii == n) mtext("Year", side = 1, line = 2)
}
invisible(dev.off())

# MP shock IRFs at three reference dates
ref_dates  <- c(1975, 1981.5, 1996)       # 1975Q1, 1981Q3, 1996Q1
ref_labels <- c("1975Q1", "1981Q3", "1996Q1")
n_ref      <- length(ref_dates)
i_mp       <- n                           # MP shock = last equation
shock      <- numeric(n); shock[i_mp] <- 1
F_sub      <- cbind(diag((p-1)*n), matrix(0, (p-1)*n, n))

store_yIR <- array(0, dim = c(nsim, n_hz, n, n_ref))
maxeig_all <- matrix(0, nsim, n_ref)
for (r in 1:n_ref) {
    t_idx <- which.min(abs(dates_q - ref_dates[r]))
    for (s in 1:nsim) {
        beta_path <- matrix(store_beta[s, ], nk, T_eff)
        beta_t    <- beta_path[, t_idx]
        A_full    <- t(matrix(beta_t, k, n))
        F <- rbind(A_full[, 2:k, drop = FALSE], F_sub)
        maxeig_all[s, r] <- max(Mod(eigen(F, only.values = TRUE)$values))
        l_path <- matrix(store_l[s, ], m, T_eff)
        l_t    <- l_path[, t_idx]
        h_t    <- store_h[s, t_idx, ]
        Lt <- diag(n); Lt[L_id] <- l_t
        Dt <- diag(exp(h_t), n, n)
        iLt   <- solve(Lt)
        Sig_t <- iLt %*% Dt %*% t(iLt)   # MATLAB's Lt \ Dt / Lt'
        store_yIR[s, , , r] <- construct_IR(beta_t, Sig_t, n_hz, shock)
    }
    cat(sprintf(paste0("Reference %s -> t = %3d : max|eig(F)| median=%.3f  ",
                       "q10=%.3f  q90=%.3f  max=%.3f\n"),
        ref_labels[r], t_idx,
        quantile(maxeig_all[, r], 0.5, type = 5),
        quantile(maxeig_all[, r], 0.1, type = 5),
        quantile(maxeig_all[, r], 0.9, type = 5),
        max(maxeig_all[, r])))
}

# normalize by posterior-median impact FFR response (one scalar per ref date)
# so the IRF corresponds to a 1pp shock; dividing each draw by its own
# impact response amplifies noise from draws with near-zero impact
for (r in 1:n_ref) {
    impact_med <- quantile(store_yIR[, 1, i_mp, r], 0.50, type = 5)
    store_yIR[, , , r] <- store_yIR[, , , r]/impact_med
}

yIR_med <- apply(store_yIR, c(2, 3, 4), quantile, probs = 0.50, type = 5)
yIR_lo  <- apply(store_yIR, c(2, 3, 4), quantile, probs = 0.16, type = 5)
yIR_hi  <- apply(store_yIR, c(2, 3, 4), quantile, probs = 0.84, type = 5)

# Figure 2: inflation and unemployment IRFs, shared y-axis per row
varnames_resp <- c("Inflation", "Unemployment")
n_resp        <- length(varnames_resp)
hz            <- 0:(n_hz-1)

y_lim <- matrix(0, n_resp, 2)
for (ii in 1:n_resp) {
    y_min <- min(yIR_lo[, ii, ])
    y_max <- max(yIR_hi[, ii, ])
    pad   <- 0.05 * (y_max - y_min)
    y_lim[ii, ] <- c(y_min - pad, y_max + pad)
}

setEPS()
postscript("Primiceri_MPshock_IR.eps", width = 8, height = 3.5)
par(mfrow = c(n_resp, n_ref), mar = c(3.5, 4, 1, 1))
for (ii in 1:n_resp) {
    for (r in 1:n_ref) {
        plot(hz, yIR_med[, ii, r], type = "n", xlim = c(-0.5, n_hz-1),
             ylim = y_lim[ii, ], xlab = "", ylab = "", bty = "n")
        plotCI(hz, yIR_lo[, ii, r], yIR_hi[, ii, r])
        lines(hz, yIR_med[, ii, r], col = "black", lwd = 1.5)
        abline(h = 0, col = "black", lwd = 0.5)
        if (r == 1) mtext(varnames_resp[ii], side = 2, line = 2.5)
        if (ii == n_resp) mtext("Quarters", side = 1, line = 2.3)
    }
}
invisible(dev.off())

# save plot data so the figures can be rebuilt without rerunning
saveRDS(list(sig_med = sig_med, sig_lo = sig_lo, sig_hi = sig_hi,
             dates_q = dates_q, yIR_med = yIR_med, yIR_lo = yIR_lo,
             yIR_hi = yIR_hi, hz = hz, ref_labels = ref_labels, i_mp = i_mp,
             n = n, p = p, k = k, nk = nk, m = m, T_eff = T_eff),
        "Primiceri_results.rds")

# summary of final results
cat("\nPosterior median residual std devs (1963Q1 / 2001Q3):\n")
for (ii in 1:n) {
    cat(sprintf("  %-16s: %.3f / %.3f\n", resnames[ii],
                sig_med[1, ii], sig_med[T_eff, ii]))
}
cat("\nMedian [16%, 84%] responses to a 1pp MP shock at horizon 8:\n")
for (ii in 1:n_resp) {
    for (r in 1:n_ref) {
        cat(sprintf("  %-12s %s: % .3f [% .3f, % .3f]\n",
                    varnames_resp[ii], ref_labels[r],
                    yIR_med[9, ii, r], yIR_lo[9, ii, r], yIR_hi[9, ii, r]))
    }
}
