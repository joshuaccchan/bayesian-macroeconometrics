# linreg_ma1_chib.R
# Computes the log marginal likelihood of the ARMA(2,1) model for US
# PCE inflation via Chib's method. The posterior ordinate at the
# posterior mean (beta*, sig2*, psi*) is factored as
#   p(beta*, sig2*, psi* | y) = p(beta* | y) * p(sig2* | y, beta*)
#                               * p(psi* | y, beta*, sig2*).
# The first factor is estimated from the main Gibbs run, the second
# from a reduced run that fixes beta at beta* and updates (sig2, psi),
# and the third by evaluating the conditional posterior of psi on a
# fine grid. The seed is reset before the reduced run so the marginal
# likelihood estimate is reproducible.

suppressMessages(library(Matrix))
source("loglike_MA1.R")
source("sample_psi_RW.R")
source("lmvnpdf.R")
source("ligampdf.R")

source("linreg_ma1.R")  # estimate the ARMA(2,1) model

nsim_re <- 5000  # size of reduced runs
nsim <- nrow(store_theta);  m <- ncol(store_theta)
T <- length(y)

theta_hat <- colMeans(store_theta)
beta_s <- theta_hat[1:(m-2)]  # beta*
sig2_s <- theta_hat[m-1]      # sig2*
psi_s <- theta_hat[m]         # psi*

# log likelihood at theta*
llike <- loglike_MA1(psi_s, y - X %*% beta_s, sig2_s)

# log prior at theta*
log_prior <- function(b, s, p) lmvnpdf(b, beta0, solve(iVbeta)) +
    ligampdf(s, nu0, S0) + log(1/2)

# 1) posterior of beta at beta_s using main run
store_lpbeta <- numeric(nsim)
for (isim in 1:nsim) {
    sig2 <- store_theta[isim, m-1]  # from the main run
    psi <- store_theta[isim, m]
    # same lower bidiagonal matrix as in linreg_ma1.R; the .m writes it
    # here as speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T)
    Hpsi <- bandSparse(T, k = c(0, -1),
                       diagonals = list(rep(1, T), rep(psi, T-1)))
    X_tilde <- as.matrix(solve(Hpsi, X));  y_tilde <- as.numeric(solve(Hpsi, y))
    Dbeta <- solve(iVbeta + crossprod(X_tilde)/sig2)
    beta_hat <- as.numeric(Dbeta %*% (iVbeta %*% beta0
        + crossprod(X_tilde, y_tilde)/sig2))
    store_lpbeta[isim] <-
        lmvnpdf(beta_s, beta_hat, Dbeta)
}
a <- max(store_lpbeta)
lpbeta <- log(mean(exp(store_lpbeta - a))) + a

# 2) posterior of sig2 at sig2_s using a reduced run
set.seed(42)  # re-seed for reproducible reduced run
beta <- beta_s  # fix beta at beta_s
e <- as.numeric(y - X %*% beta)
sig2 <- sig2_s
psi <- psi_s
store_lpsig2 <- numeric(nsim_re)
for (isim in 1:nsim_re) {
    # sample psi
    psi <- sample_psi_RW(psi, e, sig2, g_var)$psi
    Hpsi <- bandSparse(T, k = c(0, -1),
                       diagonals = list(rep(1, T), rep(psi, T-1)))
    # sample sig2
    tmp <- as.numeric(solve(Hpsi, e))
    sig2 <- 1/rgamma(1, shape = nu0 + T/2, scale = 1/(S0 + sum(tmp^2)/2))

    store_lpsig2[isim] <-
        ligampdf(sig2_s, nu0 + T/2, S0 + sum(tmp^2)/2)
}
a <- max(store_lpsig2)
lpsig2 <- log(mean(exp(store_lpsig2 - a))) + a

# 3) posterior of psi at psi_s via grid
n_grid <- 399
psi_grid <- seq(-.99, .99, length.out = n_grid)
    # ensure psi_s is on the grid
psi_grid <- sort(c(psi_grid, psi_s))
n_grid <- length(psi_grid)
idx_psi <- (psi_grid == psi_s)  # index for psi_s
lp_psi <- numeric(n_grid)  # log posterior density
for (igrid in 1:n_grid) {
    psi <- psi_grid[igrid]
    lp_psi[igrid] <- loglike_MA1(psi, y - X %*% beta_s, sig2_s)
}
p_psi <- exp(lp_psi - max(lp_psi))
# trapezoidal rule (MATLAB's trapz(psi_grid, p_psi))
p_psi <- p_psi/sum(diff(psi_grid)*(p_psi[-1] + p_psi[-n_grid])/2)
lppsi <- log(p_psi[idx_psi])

log_ml <- llike + log_prior(beta_s, sig2_s, psi_s) -
    (lpbeta + lpsig2 + lppsi)

cat(sprintf("Log marginal likelihood: %.2f\n", log_ml))
