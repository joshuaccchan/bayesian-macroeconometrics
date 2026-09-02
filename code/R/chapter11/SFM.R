# SFM.R
# Gibbs sampler for the static factor model fitted to daily exchange-rate
# returns on nine currencies. The model is
#   y_t = A f_t + eps_t,   eps_t ~ N(0, Sigma),   f_t ~ N(0, Omega),
# where A is n x r, lower triangular with ones on the diagonal, and Sigma and
# Omega are diagonal. The 3-block Gibbs sampler draws the factors f, the free
# loadings a, and the variances (Sigma, Omega). After sampling, it estimates
# the log marginal likelihood by the cross-entropy method (SFM_CE.R) and the
# variance decomposition (vardec_SFM.R). Set r to the desired number of factors.
#
# Requires: SFM_CE.R, vardec_SFM.R, logintlike_SFM.R, lmvnpdf.R, ligampdf.R

suppressMessages(library(Matrix))

source("SFM_CE.R")
source("vardec_SFM.R")
source("ligampdf.R")

# The posterior of the r = 3 static factor model is bimodal: the dominant mode
# has a log marginal likelihood of about -9699 and the minor one about -9754.
# Which mode the chain settles in is decided by the random starting value of a,
# so it is seed dependent (the Python port had to switch to seed 7). Under R's
# RNG, seed 42 - the same seed the MATLAB script uses - reaches the dominant
# mode, and so do seeds 1, 7 and 123; if a different seed is used, check that
# the reported log marginal likelihood is near -9699 and not -9754.
set.seed(42)
nsim <- 20000
burnin <- 1000
R <- 10000  # number of importance draws for the cross-entropy method
r <- 3      # # of factors;
data <- as.matrix(read.csv("daily_fx.csv", header = FALSE))
returns <- 100*log(data[2:nrow(data), ]/data[1:(nrow(data) - 1), ])
Y <- returns

T <- nrow(Y)
n <- ncol(Y)
y <- as.vector(t(Y))
na <- n*r - r*(r + 1)/2

# storage
store_a <- matrix(0, nsim, na)
store_f <- matrix(0, nsim, T*r)
store_sig2 <- matrix(0, nsim, n)
store_omega2 <- matrix(0, nsim, r)

# prior hyperparameters
a0 <- 0
Va <- 1  # aij iid N(a0,Va)
nusig2 <- 3
Ssig2 <- 1*(nusig2 - 1)*rep(1, n)
nuomega2 <- 3
Somega2 <- 1*(nuomega2 - 1)*rep(1, r)
prior <- function(ax, s, o) {
    -na/2*log(2*pi*Va) - 0.5*sum((ax - a0)^2/Va) +
        sum(ligampdf(s, nusig2, Ssig2)) + sum(ligampdf(o, nuomega2, Somega2))
}

# initialize
varY <- apply(Y, 2, var)
sig2 <- varY/2                    # diagonal elements of Sigma
omega2 <- mean(varY)/2*rep(1, r)  # diagonal elements of Omega
a <- rnorm(na)
A <- rbind(diag(r), matrix(0, n - r, r))
count_a <- 0
for (ii in 2:n) {
    nai <- min(ii - 1, r)
    A[ii, 1:nai] <- a[(count_a + 1):(count_a + nai)]
    count_a <- count_a + nai
}

tStart <- Sys.time()
for (isim in 1:(nsim + burnin)) {
    # sample f
    AiSig <- t(A) %*% Diagonal(x = 1/sig2)
    # symmpart() only removes the O(1e-16) asymmetry that sparse arithmetic can
    # leave in the r x r block, so that Kf is recognized as symmetric by chol()
    Kf <- Diagonal(x = rep(1/omega2, T)) +
        kronecker(Diagonal(T), symmpart(AiSig %*% A))
    fhat <- solve(Kf, as.numeric(kronecker(Diagonal(T), AiSig) %*% y))
    # R's chol() of a sparse symmetric matrix is the UPPER factor, so
    # solve(chol(Kf), z) is MATLAB's chol(Kf,'lower')' \ randn
    f <- as.numeric(fhat + solve(chol(Kf), rnorm(T*r)))
    F <- t(matrix(f, r, T))  # T x r - tth row is f_t

    # sample a
    count_a <- 0
    for (i in 2:n) {
        if (i <= r) {  # # of elements in ai
            nai <- i - 1
        } else {
            nai <- r
        }
        Z <- F[, 1:nai, drop = FALSE]
        Kai <- diag(1/Va, nai) + crossprod(Z)/sig2[i]
        if (i <= r) {
            ai_hat <- solve(Kai, a0/Va +
                            as.numeric(crossprod(Z, Y[, i] - F[, i]))/sig2[i])
        } else {
            ai_hat <- solve(Kai, a0/Va +
                            as.numeric(crossprod(Z, Y[, i]))/sig2[i])
        }
        ai <- as.numeric(ai_hat + solve(chol(Kai), rnorm(nai)))
        A[i, 1:nai] <- ai
        a[(count_a + 1):(count_a + nai)] <- ai
        count_a <- count_a + nai
    }

    # sample sig2 and omega2
    E <- Y - F %*% t(A)
    sig2 <- 1/rgamma(n, shape = nusig2 + T/2,
                     scale = 1/(Ssig2 + colSums(E^2)/2))
    omega2 <- 1/rgamma(r, shape = nuomega2 + T/2,
                       scale = 1/(Somega2 + colSums(F^2)/2))

    if (isim %% 5000 == 0) {
        cat(sprintf("Iteration %d of %d (%.1f%%), elapsed time: %.1f seconds\n",
                    isim, nsim + burnin, 100*isim/(nsim + burnin),
                    as.numeric(difftime(Sys.time(), tStart, units = "secs"))))
    }

    if (isim > burnin) {
        i <- isim - burnin
        store_a[i, ] <- a
        store_f[i, ] <- f
        store_sig2[i, ] <- sig2
        store_omega2[i, ] <- omega2
    }
}
a_mean <- colMeans(store_a)
astd <- apply(store_a, 2, sd)
f_mean <- t(matrix(colMeans(store_f), r, T))
sig2_mean <- colMeans(store_sig2)
omega2_mean <- colMeans(store_omega2)

CE <- SFM_CE(store_a, store_sig2, store_omega2, Y, prior, R)
cat(sprintf("Log marginal likelihood = %.1f (s.e. = %.2f)\n",
            CE$logml, CE$logml_std))

vd <- vardec_SFM(store_a, store_sig2, store_omega2)
sys_mean <- vd$sys_mean
idio_mean <- vd$idio_mean

varnames <- c("AUD", "CAD", "EUR", "JPY", "CHF", "GBP", "KRW", "NZD", "TWD")

cat("Systematic and idiosyncratic variance shares:\n")
for (i in 1:n) {
    cat(sprintf("%s: systematic = %.2f, idiosyncratic = %.2f\n",
                varnames[i], sys_mean[i], idio_mean[i]))
}
