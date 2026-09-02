# DFM.R
# Gibbs sampler for the dynamic factor model fitted to a large monthly
# macroeconomic panel (FRED-MD). The model is
#   y_t = A f_t + eps_t,                              eps_t ~ N(0, Sigma),
#   f_t = Phi_1 f_{t-1} + ... + Phi_p f_{t-p} + u_t,  u_t   ~ N(0, Omega),
# with A lower triangular (ones on the diagonal) and Sigma, Omega diagonal.
# This script estimates the one-factor, one-lag specification (r=1, p=1) as a
# business-cycle indicator. The 4-block Gibbs sampler draws the factor path f,
# the free loadings a, the variances (Sigma, Omega), and the AR coefficients
# phi (drawn from a normal truncated to the stationary region). It plots the
# posterior mean factor with NBER recession shading.
#
# Requires: shade_nber_recessions.R

suppressMessages(library(Matrix))

source("shade_nber_recessions.R")

set.seed(42)  # for reproducibility

nsim <- 20000
burnin <- 1000
r <- 1  # number of factors

# load data
raw <- read.csv("FRED-MD.csv", check.names = FALSE)
dates_num <- raw[, 1]  # first column contains dates as year + month/12
month_idx <- round(12*dates_num)  # months since year 0
year_part <- floor((month_idx - 1)/12)
month_part <- month_idx - 12*year_part
dates <- as.Date(sprintf("%d-%02d-01", year_part, month_part))
data <- as.matrix(raw[, 2:ncol(raw)])  # remaining columns are data
varnames <- colnames(raw)[2:ncol(raw)]

# move the 6th column (INDPRO) to the first column
perm <- c(6, 1:5, 7:ncol(data))
data <- data[, perm]
varnames <- varnames[perm]

# remove columns with missing values
idx <- apply(!is.na(data), 2, all)
data <- data[, idx]
varnames <- varnames[idx]

# standardize the data
data_mean <- colMeans(data)
data_std <- apply(data, 2, sd)
Y <- sweep(sweep(data, 2, data_mean, "-"), 2, data_std, "/")

T <- nrow(Y)
n <- ncol(Y)
y <- as.vector(t(Y))

# storage
store_F <- array(0, c(nsim, T, r))
store_A <- array(0, c(nsim, n, r))
store_sig2 <- matrix(0, nsim, n)
store_omega2 <- matrix(0, nsim, r)
store_phi <- matrix(0, nsim, r)

# prior hyperparameters
a0 <- 0
Va <- 1  # a_ij iid N(a0,Va)
phi0 <- numeric(r)
Vphi <- rep(1, r)
nusig2 <- 3
Ssig2 <- (nusig2 - 1)*rep(1, n)
nuomega2 <- 3
Somega2 <- (nuomega2 - 1)*rep(1, r)

# initialize the Markov chain
sig2 <- apply(Y, 2, var)
omega2 <- rep(1, r)
phi <- 0.5*rep(1, r)
Phi <- Diagonal(x = phi)
A <- rbind(diag(r), matrix(0, n - r, r))  # lower-triangular normalization

# matrices used to build H_phi
hzeros <- sparseMatrix(i = integer(0), j = integer(0), dims = c(r, (T - 1)*r))
vzeros <- sparseMatrix(i = integer(0), j = integer(0), dims = c(T*r, r))
HPhi <- Diagonal(T*r) -
    cbind(rbind(hzeros, kronecker(Diagonal(T - 1), Phi)), vzeros)

for (isim in 1:(nsim + burnin)) {
    # sample f
    iSig <- Diagonal(x = 1/sig2)
    iOmega <- Diagonal(x = rep(1/omega2, T))
    # symmpart() only removes the O(1e-16) asymmetry that sparse arithmetic can
    # leave in the r x r block, so that Kf is recognized as symmetric by chol()
    Kf <- crossprod(HPhi, iOmega %*% HPhi) +
        kronecker(Diagonal(T), symmpart(t(A) %*% iSig %*% A))
    f_hat <- solve(Kf, as.numeric(kronecker(Diagonal(T), t(A) %*% iSig) %*% y))
    # R's chol() of a sparse symmetric matrix is the UPPER factor, so
    # solve(chol(Kf), z) is MATLAB's chol(Kf,'lower')' \ randn
    f <- as.numeric(f_hat + solve(chol(Kf), rnorm(T*r)))
    F <- t(matrix(f, r, T))

    # sample A equation by equation
    for (i in 2:n) {
        nai <- min(i - 1, r)
        Xf <- F[, 1:nai, drop = FALSE]
        K_ai <- diag(1/Va, nai) + crossprod(Xf)/sig2[i]
        if (i <= r) {
            ai_hat <- solve(K_ai, (a0/Va)*rep(1, nai) +
                            as.numeric(crossprod(Xf, Y[, i] - F[, i]))/sig2[i])
        } else {
            ai_hat <- solve(K_ai, (a0/Va)*rep(1, nai) +
                            as.numeric(crossprod(Xf, Y[, i]))/sig2[i])
        }
        A[i, 1:nai] <- ai_hat + solve(chol(K_ai), rnorm(nai))
    }

    # sample sig2
    E_y <- Y - F %*% t(A)
    sig2 <- 1/rgamma(n, shape = nusig2 + T/2,
                     scale = 1/(Ssig2 + colSums(E_y^2)/2))

    # sample omega2
    E_f <- rbind(F[1, , drop = FALSE],
                 F[2:T, , drop = FALSE] -
                     as.matrix(F[1:(T - 1), , drop = FALSE] %*% Phi))
    omega2 <- 1/rgamma(r, shape = nuomega2 + T/2,
                       scale = 1/(Somega2 + colSums(E_f^2)/2))

    # sample phi equation by equation (normal truncated to |phi|<1)
    Zf <- rbind(matrix(0, 1, r), F[1:(T - 1), , drop = FALSE])
    for (jj in 1:r) {
        Kphi_j <- 1/Vphi[jj] + sum(Zf[, jj]^2)/omega2[jj]
        phi_hat_j <- (phi0[jj]/Vphi[jj] +
                          sum(Zf[, jj]*F[, jj])/omega2[jj])/Kphi_j
        phi_sd_j <- sqrt(1/Kphi_j)
        accepted <- FALSE
        while (!accepted) {
            phi_prop <- phi_hat_j + phi_sd_j*rnorm(1)
            if (abs(phi_prop) < 1) {
                phi[jj] <- phi_prop
                accepted <- TRUE
            }
        }
    }
    Phi <- Diagonal(x = phi)
    HPhi <- Diagonal(T*r) -
        cbind(rbind(hzeros, kronecker(Diagonal(T - 1), Phi)), vzeros)

    if (isim > burnin) {
        isave <- isim - burnin
        store_F[isave, , ] <- F
        store_A[isave, , ] <- A
        store_sig2[isave, ] <- sig2
        store_omega2[isave, ] <- omega2
        store_phi[isave, ] <- phi
    }

    if (isim %% 5000 == 0) {
        cat(sprintf("Iteration %d of %d (%.1f%%)\n",
                    isim, nsim + burnin, 100*isim/(nsim + burnin)))
    }
}
F_mean <- matrix(apply(store_F, c(2, 3), mean), T, r)

# posterior summary of the AR coefficient
phi_mean <- colMeans(store_phi)
phi_q <- apply(store_phi, 2, quantile, probs = c(0.025, 0.975), type = 5)
cat(sprintf("Posterior mean of phi = %.3f, 95%% CI = (%.3f, %.3f)\n",
            phi_mean[1], phi_q[1, 1], phi_q[2, 1]))

# decimal-year time axis for plotting
tt <- year_part + (month_part - 1)/12

# full-sample and pre-COVID plots of the posterior mean factor
idx_pre <- tt <= 2019 + 11/12  # through December 2019
tt_pre_end <- tt[max(which(idx_pre))]

setEPS()
postscript("DFM_f.eps", width = 9, height = 6)
par(mfrow = c(2, 1), mar = c(3, 3, 1, 1), mgp = c(2, 0.7, 0), cex.axis = 1.1)

# top panel: full sample
yl_full <- c(min(F_mean[, 1]) - 0.2, max(F_mean[, 1]) + 0.2)
plot(tt, F_mean[, 1], type = "n", xlim = c(tt[1], tt[T]), ylim = yl_full,
     xlab = "", ylab = "", xaxs = "i", yaxs = "i")
shade_nber_recessions(yl_full[1], yl_full[2])
abline(h = 0, col = "black")
lines(tt, F_mean[, 1], col = "black", lwd = 1.5)
box()

# bottom panel: pre-COVID sample
yl_pre <- c(min(F_mean[idx_pre, 1]) - 0.2, max(F_mean[idx_pre, 1]) + 0.2)
plot(tt[idx_pre], F_mean[idx_pre, 1], type = "n",
     xlim = c(tt[1], tt_pre_end), ylim = yl_pre,
     xlab = "", ylab = "", xaxs = "i", yaxs = "i")
shade_nber_recessions(yl_pre[1], yl_pre[2])
abline(h = 0, col = "black")
lines(tt[idx_pre], F_mean[idx_pre, 1], col = "black", lwd = 1.5)
box()

invisible(dev.off())
