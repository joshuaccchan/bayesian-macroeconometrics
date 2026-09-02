# infmix_logchi2.R
# Dirichlet process mixture (DPM) of normals for the log-chi^2 density,
# estimated with a collapsed Gibbs sampler.
# Requires: tpdfLS.R

source("tpdfLS.R")

nsim   <- 10000
burnin <- 1000

# generate data
set.seed(1)
T  <- 2000
df <- 1
y  <- log(rchisq(T, df))

# true log-chi^2_1 density
ftrue <- function(x) 1/sqrt(2*pi) * exp(0.5*x - 0.5*exp(x))

# prior hyperparameters
nu0 <- 2;  S0  <- 1
mu0 <- 0;  Vmu <- 100
a_alpha <- 1; b_alpha <- 1  # prior on alpha

# grid
ngrid <- 500
xgrid <- seq(-10, 5, length.out = ngrid)
store_mixden <- matrix(0, nsim, ngrid)
store_M <- numeric(nsim)

# machine constants used to avoid log(0), as in the MATLAB version
eps_    <- .Machine$double.eps    # MATLAB eps
realmin <- .Machine$double.xmin   # MATLAB realmin

local_pred_t <- function(z, Tm, sumy, sumy2, mu0, Vmu, nu0, S0) {
    # Posterior predictive density of a normal-inverse-gamma component
    # evaluated at z. Integrating out (mu, sig2) under the conjugate base
    # measure G_0 yields a Student-t; setting Tm = 0 returns the prior
    # predictive f_0 under G_0.
    #
    # Inputs:
    #   z    : scalar or vector of evaluation points (e.g. the grid)
    #   Tm   : number of observations currently in the cluster
    #   sumy : sum of the cluster observations, sum_t y_t
    #   sumy2: sum of squared cluster observations, sum_t y_t^2
    #   mu0  : prior mean of mu
    #   Vmu  : prior variance scale, mu | sig2 ~ N(mu0, sig2*Vmu)
    #   nu0  : prior shape for sig2, sig2 ~ IG(nu0, S0)
    #   S0   : prior scale for sig2
    #
    # Output:
    #   f    : predictive density evaluated at z (same size as z)
    Kmu    <- 1/Vmu + Tm
    nu_hat <- nu0 + Tm/2
    if (Tm == 0) {
        mu_hat <- mu0
        S_hat  <- S0
    } else {
        mu_hat <- (mu0/Vmu + sumy)/Kmu
        S_hat  <- S0 + 0.5*(sumy2 + mu0^2/Vmu - Kmu*mu_hat^2)
    }
    s2 <- (S_hat/nu_hat)*(1 + 1/Kmu)
    nu <- 2*nu_hat

    tpdfLS(z, mu_hat, s2, nu)
}

# posterior predictive density
pred_t_density <- function(z, Tm, sy, sy2)
    local_pred_t(z, Tm, sy, sy2, mu0, Vmu, nu0, S0)

# initialize using K-means. R's stats::kmeans has no kmeans++ start (MATLAB's
# default), so it is run with several random starts instead; this only seeds
# the Markov chain and does not affect the stationary distribution.
Minit <- 10
s <- kmeans(y, centers = Minit, nstart = 10, iter.max = 100)$cluster
alpha <- rgamma(1, shape = a_alpha, scale = 1/b_alpha)

# maintain clusters dynamically using sufficient statistics
# for cluster m: Tm(m), sumy(m), sumy2(m)
M <- max(s)
Tm <- numeric(M)
sumy <- numeric(M)
sumy2 <- numeric(M)

for (m in 1:M) {
    idx <- (s == m)
    Tm[m] <- sum(idx)
    ym <- y[idx]
    sumy[m] <- sum(ym)
    sumy2[m] <- sum(ym*ym)
}

for (isim in 1:(nsim + burnin)) {
    # sample alpha
    M <- length(Tm)   # current # of clusters
    eta <- rbeta(1, alpha + 1, T)
    pi_eta <- (a_alpha + M - 1)/(T*(b_alpha - log(eta)) + a_alpha + M - 1)
    if (runif(1) < pi_eta) {
        alpha <- rgamma(1, shape = a_alpha + M,
                        scale = 1/(b_alpha - log(eta)))
    } else {
        alpha <- rgamma(1, shape = a_alpha + M - 1,
                        scale = 1/(b_alpha - log(eta)))
    }

    # sample s_t
    for (t in 1:T) {
        M <- length(Tm)
        mt <- s[t]
        # remove observation t from its current cluster
        Tm[mt] <- Tm[mt] - 1
        sumy[mt] <- sumy[mt] - y[t]
        sumy2[mt] <- sumy2[mt] - y[t]^2

        # if cluster becomes empty, remove it
        if (Tm[mt] == 0) {
            # move last cluster to mt (if mt not last)
            if (mt != M) {
                Tm[mt] <- Tm[M]
                sumy[mt] <- sumy[M]
                sumy2[mt] <- sumy2[M]

                # relabel all obs. assigned to M to mt
                s[s == M] <- mt
            }
            # shrink arrays
            Tm <- Tm[-M]; sumy <- sumy[-M]; sumy2 <- sumy2[-M]
            M <- M - 1
        }

        # compute assignment probabilities
        logp <- numeric(M + 1)
        for (m in seq_len(M)) {
            fm <- pred_t_density(y[t], Tm[m], sumy[m], sumy2[m])
            # (eps, realmin) added to avoid log(0)
            logp[m] <- log(Tm[m] + eps_) + log(fm + realmin)
        }

        # prior predictive (new cluster) density f0(y_t)
        f0 <- pred_t_density(y[t], 0, 0, 0)
        logp[M + 1] <- log(alpha + eps_) + log(f0 + realmin)

        # normalize safely
        logp <- logp - max(logp)
        p <- exp(logp)
        p <- p/sum(p)

        # sample new assignment
        newm <- which(runif(1) <= cumsum(p))[1]

        if (newm <= M) {
            # assign to existing cluster
            s[t] <- newm
            Tm[newm] <- Tm[newm] + 1
            sumy[newm] <- sumy[newm] + y[t]
            sumy2[newm] <- sumy2[newm] + y[t]^2
        } else {
            # create new cluster
            M <- M + 1
            s[t] <- M
            Tm <- c(Tm, 1)
            sumy <- c(sumy, y[t])
            sumy2 <- c(sumy2, y[t]^2)
        }
    }

    if (isim > burnin) {
        isave <- isim - burnin
        # posterior predictive
        f0g <- pred_t_density(xgrid, 0, 0, 0)
        mixden <- (alpha/(alpha + T)) * f0g
        for (m in seq_len(M)) {
            fm_g <- pred_t_density(xgrid, Tm[m], sumy[m], sumy2[m])
            mixden <- mixden + (Tm[m]/(alpha + T)) * fm_g
        }
        store_mixden[isave, ] <- mixden
        store_M[isave] <- M
    }
}

# posterior summaries
M_mean <- mean(store_M)
mixden_mean <- colMeans(store_mixden)
mixden_low  <- apply(store_mixden, 2, quantile, probs = 0.025, type = 5)
mixden_high <- apply(store_mixden, 2, quantile, probs = 0.975, type = 5)

cat(sprintf("posterior mean # of clusters: %.2f\n", M_mean))
cat(sprintf(paste0("max abs deviation of the DPM density estimate from the ",
                   "true log-chi^2_1 density: %.4f\n"),
            max(abs(mixden_mean - ftrue(xgrid)))))

# plot
setEPS()
postscript("infmixture_logchi2.eps", width = 6.4, height = 4.8)
par(mar = c(4.2, 4.2, 1, 1), bty = "l", las = 1, cex.axis = 1.15, cex.lab = 1.15)
plot(NA, xlim = c(-10, 5), ylim = c(0, max(mixden_high, ftrue(xgrid))),
     xaxs = "i", xlab = expression(y), ylab = "Density")

polygon(c(xgrid, rev(xgrid)), c(mixden_low, rev(mixden_high)),
        col = gray(0.85), border = NA)

lines(xgrid, mixden_mean, col = "black", lty = 1, lwd = 2)
lines(xgrid, ftrue(xgrid), col = "black", lty = 2, lwd = 2)

legend("topleft", legend = c("DPM", "true density"), lty = c(1, 2),
       lwd = 2, col = "black", bty = "n", cex = 1.15)

# Save as EPS
invisible(dev.off())
