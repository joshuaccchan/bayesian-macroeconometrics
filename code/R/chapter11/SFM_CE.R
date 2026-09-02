# SFM_CE.R
# Estimates the log marginal likelihood of the static factor model using the
# cross-entropy method. The importance density is a product of a Gaussian
# for the free loadings and inverse-gamma densities for the variances,
# with parameters matched to the posterior draws. The point estimate and
# its numerical standard error are obtained from 20 importance batches.
#
# Requires: logintlike_SFM.R, lmvnpdf.R, ligampdf.R
#
# Inputs:
#   store_a     : posterior draws of the free loadings, nsim x na
#   store_Sig   : posterior draws of the idiosyncratic variances, nsim x n
#   store_Omega : posterior draws of the factor variances, nsim x r
#   Y           : data, T x n
#   prior       : function evaluating the log prior density
#   R           : number of importance samples
#
# Outputs (returned in a list):
#   logml     : estimated log marginal likelihood
#   logml_std : numerical standard error based on 20 importance batches

source("logintlike_SFM.R")
source("lmvnpdf.R")
source("ligampdf.R")

# MATLAB's gamfit(x) returns the maximum likelihood estimates (shape, scale) of
# the gamma density. R has no base-package equivalent (MASS::fitdistr is not a
# permitted dependency), so the MLE is computed here directly: the shape solves
#   log(a) - digamma(a) = log(mean(x)) - mean(log(x)),
# which is solved by Newton's method from Minka's starting value, and the scale
# is then mean(x)/a.
gamfit <- function(x) {
    s <- log(mean(x)) - mean(log(x))
    a <- (3 - s + sqrt((s - 3)^2 + 24*s))/(12*s)  # starting value
    for (it in 1:200) {
        a_new <- a - (log(a) - digamma(a) - s)/(1/a - trigamma(a))
        if (!is.finite(a_new) || a_new <= 0) break
        if (abs(a_new - a) < 1e-12*a) {
            a <- a_new
            break
        }
        a <- a_new
    }
    c(a, mean(x)/a)  # (shape, scale)
}

SFM_CE <- function(store_a, store_Sig, store_Omega, Y, prior, R) {
    R <- 20*ceiling(R/20)  # make R divisible by 20 for batching
    na <- ncol(store_a)
    n <- ncol(Y)
    r <- ncol(store_Omega)

    # estimate the parameters of the importance density
    a_bar <- colMeans(store_a)
    Da_bar <- cov(store_a)
    Da_bar <- Da_bar + 1e-10*diag(na)  # small ridge for numerical stability
    CDa_bar <- t(chol(Da_bar))         # lower-triangular factor

    nusig2_bar <- numeric(n)
    Ssig2_bar <- numeric(n)
    for (ii in 1:n) {
        tmp <- gamfit(1/store_Sig[, ii])
        nusig2_bar[ii] <- tmp[1]
        Ssig2_bar[ii] <- 1/tmp[2]
    }

    nuomega2_bar <- numeric(r)
    Somega2_bar <- numeric(r)
    for (jj in 1:r) {
        tmp <- gamfit(1/store_Omega[, jj])
        nuomega2_bar[jj] <- tmp[1]
        Somega2_bar[jj] <- 1/tmp[2]
    }

    # draw from the importance density
    a_IS <- matrix(a_bar, R, na, byrow = TRUE) +
        t(CDa_bar %*% matrix(rnorm(na*R), na, R))

    Sig_IS <- matrix(0, R, n)
    for (ii in 1:n) {
        Sig_IS[, ii] <- 1/rgamma(R, shape = nusig2_bar[ii],
                                 scale = 1/Ssig2_bar[ii])
    }

    Omega_IS <- matrix(0, R, r)
    for (jj in 1:r) {
        Omega_IS[, jj] <- 1/rgamma(R, shape = nuomega2_bar[jj],
                                   scale = 1/Somega2_bar[jj])
    }

    # log importance density
    g_IS <- function(ax, s, o) {
        lmvnpdf(ax, a_bar, Da_bar) +
            sum(ligampdf(s, nusig2_bar, Ssig2_bar)) +
            sum(ligampdf(o, nuomega2_bar, Somega2_bar))
    }

    # log importance weights
    store_w <- numeric(R)
    for (isim in 1:R) {
        a <- a_IS[isim, ]
        Sig <- Sig_IS[isim, ]
        Omega <- Omega_IS[isim, ]

        llike <- logintlike_SFM(Y, a, Sig, Omega)
        store_w[isim] <- llike + prior(a, Sig, Omega) - g_IS(a, Sig, Omega)
    }

    # batch estimate of log marginal likelihood
    shortw <- matrix(store_w, R/20, 20)          # 20 batches
    maxw <- apply(shortw, 2, max)                # batch-specific maxima
    bigml <- log(colMeans(exp(sweep(shortw, 2, maxw, "-")))) + maxw

    logml <- mean(bigml)
    logml_std <- sd(bigml)/sqrt(20)

    list(logml = logml, logml_std = logml_std)
}
