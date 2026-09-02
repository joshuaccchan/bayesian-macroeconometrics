# VAR_MASV_sol.R
# Solution to the exercise "Moving Average Stochastic Volatility", part (c):
# a Bayesian VAR whose reduced-form errors follow a common MA(1) with a common
# stochastic-volatility innovation:
#     y_t   = A'x_t + eps_t,
#     eps_t = v_t + psi_1 v_{t-1},          |psi_1| < 1,
#     (v_t | Sig,h_t) ~ N(0, exp(h_t) Sig),
#     h_t   = phi h_{t-1} + u_t^h,   u_t^h ~ N(0,sigh2),  h_1 ~ N(0,sigh2/(1-phi^2)).
#
# Stacking the T observations (v_0 = 0) gives U = M_psi V, with M_psi the T x T
# unit lower-bidiagonal MA operator (1 on the diagonal, psi_1 on the first
# subdiagonal) and V the T x n whitened innovations (rows v_t'). Hence
# vec(U) ~ N(0, Sig kron Omega) with Omega = M_psi diag(exp(h)) M_psi' the
# symmetric tridiagonal (banded) matrix with
#     Omega_11   = exp(h_1),
#     Omega_tt   = exp(h_t) + psi_1^2 exp(h_{t-1}),  t >= 2,
#     Omega_{t,t-1} = psi_1 exp(h_{t-1}).
# Conditional on Omega, (A,Sig) has the natural-conjugate NIW posterior of
# Theorem 15.1 (thm:largeVAR-cond-NIW). All T x T operations use the sparse
# banded Omega and its Cholesky -- no dense T x T inverses are formed.
#
# Gibbs blocks:
#   1. (A,Sig | Y,Omega): natural-conjugate NIW.
#   2. (psi_1 | Y,A,Sig,h): random-walk Metropolis-Hastings on (-1,1). Because
#      |M_psi|^n = 1, the log-conditional is
#      -0.5 sum_t exp(-h_t) v_t(psi_1)' Sig^{-1} v_t(psi_1) + log prior(psi_1),
#      with V(psi_1) = M_psi \ U obtained by a bidiagonal solve.
#   3. (h | Y,A,Sig,psi_1,phi,sigh2): whiten V = M_psi \ U so v_t ~ N(0,e^{h_t}Sig)
#      is a common-SV structure, then the same ARMH update as VAR_CSV_o.R.
#   4. (phi,sigh2 | h): exactly as in VAR_CSV_o.R.
# The outlier component of VAR_CSV_o.R is omitted here for simplicity.
#
# Requires Minn_NCP.R, sample_CSV_h_ARMH.R, SV_RW_gaussian_approx.R,
# shaded_band.R, and inefficiency_factor.R (with specvar0.R).

suppressMessages(library(Matrix))
source("Minn_NCP.R")
source("sample_CSV_h_ARMH.R")
source("SV_RW_gaussian_approx.R")
source("shaded_band.R")
source("specvar0.R")
source("inefficiency_factor.R")

set.seed(42)

# MATLAB's iwishrnd (Statistics Toolbox) has no base-R counterpart: draw
# Sig ~ IW(S, df) from the Bartlett decomposition of a Wishart(df, S^{-1}).
iwishrnd <- function(S, df) {
    n <- nrow(S)
    C <- chol(S)                    # upper: S = t(C) %*% C
    A <- matrix(0, n, n)            # lower-triangular Bartlett factor
    diag(A) <- sqrt(rchisq(n, df - (1:n) + 1))
    A[lower.tri(A)] <- rnorm(n*(n-1)/2)
    crossprod(forwardsolve(A, C))   # = (A^{-1} C)' (A^{-1} C)
}

p <- 12            # number of lags (monthly data)
nsims <- 10000
burnin <- 2000
sig_psi <- 0.03    # random-walk MH proposal std for psi_1

# ---- load FRED-MD data (same 25-variable panel as VAR_CSV_o.R) ----
raw <- read.csv("FRED-MD.csv", check.names = FALSE)
vars <- c("RPI", "DPCERA3M086SBEA", "INDPRO", "CUMFNS", "UNRATE", "PAYEMS",
          "CES0600000007", "CES0600000008", "WPSFD49207", "PCEPI", "HOUST",
          "S&P 500", "EXUSUKx", "GS5", "GS10", "BAAFFM",   # Carriero et al. (2024)
          "FEDFUNDS", "TB3MS", "GS1", "M2SL", "BUSLOANS", "CPIAUCSL",
          "OILPRICEx", "RETAILx", "PERMIT")                # additional series
data <- as.matrix(raw[, vars])
dates <- raw[[1]]
ok <- rowSums(is.na(data)) == 0
data <- data[ok, , drop = FALSE]; dates <- dates[ok]
Y0 <- data[1:p, , drop = FALSE]
Y  <- data[(p+1):nrow(data), , drop = FALSE]
dates <- dates[(p+1):length(dates)]
T <- nrow(Y)
n <- ncol(Y)
k <- n*p + 1
cat(sprintf("FRED-MD: n=%d variables, sample %.2f-%.2f (%d months)\n",
            length(vars), dates[1], dates[length(dates)], T))

# ---- prior hyperparameters ----
kappa1 <- 100; kappa2 <- 0.2^2; rw <- 0
prior <- Minn_NCP(Y, Y0, p, kappa1, kappa2, rw)
A0 <- prior$A0; VA <- prior$VA; nu0 <- prior$nu0; S0 <- prior$S0
iva <- 1/diag(VA)                 # diagonal of VA^{-1} (sparse iVA in the .m)
phi0 <- 0.95; Vphi <- 0.1^2       # common SV: phi ~ N(phi0,Vphi) on (-1,1)
nuh0 <- 20; Sh0 <- 0.05*(nuh0-1)  # sigh2 ~ IG(nuh0,Sh0)
# psi_1 ~ U(-1,1): flat prior, so it drops out of the MH ratio

# ---- regressor matrix Z = [1, y_{t-1}',...,y_{t-p}'] ----
tmpY <- rbind(Y0[(nrow(Y0)-p+1):nrow(Y0), , drop = FALSE], Y)
Z <- matrix(0, T, n*p)
for (i in 1:p) {
    Z[, ((i-1)*n+1):(i*n)] <- tmpY[(p-i+1):(nrow(tmpY)-i), , drop = FALSE]
}
Z <- cbind(1, Z)

# ---- storage ----
store_A <- matrix(0, k, n)
store_Sig <- matrix(0, n, n)
store_h <- matrix(0, nsims, T)
store_psi <- numeric(nsims)
store_theta <- matrix(0, nsims, 2)   # [phi, sigh2]
counth <- 0; countphi <- 0; countpsi <- 0

# ---- initialize the Markov chain ----
phi <- phi0; sigh2 <- 0.1; psi_1 <- 0
KA_ols <- crossprod(Z); diag(KA_ols) <- diag(KA_ols) + iva
A_ols <- solve(KA_ols, crossprod(Z, Y))
U_ols <- Y - Z %*% A_ols
C_ols <- t(chol(crossprod(U_ols)/T))              # lower factor
s2_init <- rowSums(t(forwardsolve(C_ols, t(U_ols)))^2)
h <- SV_RW_gaussian_approx(s2_init, mean(log(s2_init)), sigh2)
h <- h - mean(h)

# fixed index vectors for building the sparse tridiagonal Omega each sweep
ii <- c(1:T, 1:(T-1), 2:T)
jj <- c(1:T, 2:T,     1:(T-1))
# fixed subdiagonal pattern for the MA operator M_psi
Msub <- sparseMatrix(i = 2:T, j = 1:(T-1), x = 1, dims = c(T, T))

cat("Starting MCMC for the VAR with common MA(1)-SV errors....\n")
t_start <- Sys.time()
for (isim in 1:(nsims+burnin)) {
    eh <- exp(h)

    # ---- sparse tridiagonal Omega = M_psi diag(exp(h)) M_psi' ----
    md <- eh; md[2:T] <- eh[2:T] + psi_1^2*eh[1:(T-1)]   # main diagonal
    od <- psi_1*eh[1:(T-1)]                              # first off-diagonal
    Om <- forceSymmetric(sparseMatrix(i = ii, j = jj, x = c(md, od, od),
                                      dims = c(T, T)))
    COm <- chol(Om)                   # sparse UPPER bidiagonal Cholesky factor
    OmiZ <- as.matrix(solve(COm, solve(t(COm), Z)))   # Omega^{-1} Z, banded solves
    OmiY <- as.matrix(solve(COm, solve(t(COm), Y)))   # Omega^{-1} Y, banded solves

    # ---- block 1: (A,Sig | Y,Omega) natural-conjugate NIW ----
    KA <- crossprod(Z, OmiZ)
    diag(KA) <- diag(KA) + iva
    KA <- (KA + t(KA))/2
    # CKA is the UPPER factor (MATLAB stores the lower one): KA = t(CKA)%*%CKA;
    # backsolve(CKA, x, transpose = TRUE) is MATLAB's CKA_lower \ x
    CKA <- chol(KA)
    Ahat <- backsolve(CKA, backsolve(CKA, iva*A0 + crossprod(Z, OmiY),
                                     transpose = TRUE))
    Shat <- S0 + crossprod(A0, iva*A0) + crossprod(Y, OmiY) -
            crossprod(Ahat, KA %*% Ahat)
    Shat <- (Shat + t(Shat))/2
    Sig <- iwishrnd(Shat, nu0+T)
    CSig <- t(chol(Sig))              # lower factor
    A <- Ahat + backsolve(CKA, matrix(rnorm(k*n), k, n)) %*% t(CSig)

    U <- Y - Z %*% A   # reduced-form residuals (rows eps_t')

    # ---- block 2: psi_1 via random-walk MH on (-1,1) (flat prior) ----
    Mpsi <- Diagonal(T) + psi_1*Msub
    V <- as.matrix(solve(Mpsi, U))          # whitened innovations, rows v_t'
    W <- t(forwardsolve(CSig, t(V)))        # v_t standardized by chol(Sig)
    llik <- -0.5*sum(exp(-h)*rowSums(W^2))
    psic <- psi_1 + sig_psi*rnorm(1)
    if (abs(psic) < 1) {
        Vc <- as.matrix(solve(Diagonal(T) + psic*Msub, U))
        Wc <- t(forwardsolve(CSig, t(Vc)))
        llikc <- -0.5*sum(exp(-h)*rowSums(Wc^2))
        if (log(runif(1)) < llikc - llik) {
            psi_1 <- psic
            countpsi <- countpsi + 1
            W <- Wc
        }
    }

    # ---- block 3: common log-volatility h via the Laplace-based ARMH ----
    s2 <- rowSums(W^2)                  # per-period sum of squares over n vars
    out <- sample_CSV_h_ARMH(s2, phi, sigh2, h, n, 30)
    h <- out$h; flag <- out$accept
    counth <- counth + flag
    h <- h - mean(h)                    # common SV normalized to zero mean

    # ---- block 4: sigh2 and phi (exactly as in VAR_CSV_o.R) ----
    eh_ar <- c(h[1]*sqrt(1-phi^2), h[2:T] - phi*h[1:(T-1)])
    # R's rgamma takes shape and scale
    sigh2 <- 1/rgamma(1, shape = nuh0+T/2, scale = 1/(Sh0 + sum(eh_ar^2)/2))
    Kphi <- 1/Vphi + sum(h[1:(T-1)]^2)/sigh2
    phihat <- (phi0/Vphi + sum(h[1:(T-1)]*h[2:T])/sigh2)/Kphi
    phic <- phihat + rnorm(1)/sqrt(Kphi)
    gphi <- function(x) -.5*log(sigh2/(1-x^2)) - .5*(1-x^2)/sigh2*h[1]^2
    if (abs(phic) < .9999) {
        if (exp(gphi(phic)-gphi(phi)) > runif(1)) {
            phi <- phic; countphi <- countphi + 1
        }
    }

    if (isim > burnin) {
        isave <- isim - burnin
        store_A <- store_A + A
        store_Sig <- store_Sig + Sig
        store_h[isave, ] <- h
        store_psi[isave] <- psi_1
        store_theta[isave, ] <- c(phi, sigh2)
    }

    if (isim %% 1000 == 0) {
        cat(isim, "loops... \n")
    }
}
cat(sprintf("MCMC takes %.1f seconds\n",
            as.numeric(difftime(Sys.time(), t_start, units = "secs"))))

# ---- posterior summaries ----
store_h <- store_h - rowMeans(store_h)   # level of h not separately identified
A_mean <- store_A/nsims
Sig_mean <- store_Sig/nsims
theta_mean <- colMeans(store_theta)
h_mean <- colMeans(store_h)
vol <- exp(store_h/2)                    # e^{h_t/2}
vol_mean <- colMeans(vol)
vol_ci <- t(apply(vol, 2, quantile, probs = c(.05, .95), type = 5))

psi_mean <- mean(store_psi)
psi_ci <- quantile(store_psi, probs = c(.05, .95), type = 5)
cat(sprintf("psi_1 posterior mean %.4f, 90%% CI [%.4f, %.4f]\n",
            psi_mean, psi_ci[1], psi_ci[2]))
cat(sprintf("psi_1 MH acceptance rate: %.3f\n", countpsi/(nsims+burnin)))
cat(sprintf("acceptance rate (h): %.3f ; (phi): %.3f\n",
            counth/(nsims+burnin), countphi/(nsims+burnin)))
cat(sprintf("phi posterior mean %.3f ; sigh2 posterior mean %.4f\n",
            theta_mean[1], theta_mean[2]))

IF_psi <- inefficiency_factor(store_psi, 500)
IF_h <- inefficiency_factor(store_h, 500)
cat(sprintf("IF of psi_1: %.1f ; IF of h: mean %.1f, median %.1f\n",
            IF_psi, mean(IF_h), median(IF_h)))

# ---- figures (saved as vector PDFs to the writeup figures folder) ----
figdir <- "figures"   # output folder (was an absolute path in the author's setup)
if (!dir.exists(figdir)) dir.create(figdir)

# (i) common volatility e^{h_t/2} with 90% credible band
pdf(file.path(figdir, "sol_largeVAR_MASV_h.pdf"), width = 8, height = 3)
plot(dates, vol_mean, type = "n", ylim = range(vol_ci),
     xlab = "", ylab = "", bty = "n", cex.axis = 1.2)
shaded_band(dates, vol_ci[, 1], vol_ci[, 2])
lines(dates, vol_mean, col = "black")
invisible(dev.off())

# (ii) histogram of the posterior draws of psi_1
pdf(file.path(figdir, "sol_largeVAR_MASV_psi.pdf"), width = 5, height = 3.5)
hist(store_psi, breaks = 50, freq = FALSE, col = gray(0.7), border = NA,
     main = "", xlab = expression(psi[1]), cex.axis = 1.2, cex.lab = 1.2)
invisible(dev.off())

cat(sprintf("figures written to %s\n", figdir))
