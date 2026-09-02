# VAR_CSV_o.R
# Metropolis-within-Gibbs sampler for a Bayesian VAR with common stochastic
# volatility and an outlier component. The VAR coefficients A and the
# covariance Sig have a natural conjugate prior.
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

p <- 12         # number of lags (monthly data)
nsims <- 30000
burnin <- 2000

# load monthly FRED-MD data: the 16 variables of Carriero, Clark, Marcellino
# and Mertens (2024), augmented with nine series spanning interest rates,
# money, credit, and prices for a 25-variable panel
raw <- read.csv("FRED-MD.csv", check.names = FALSE)
vars <- c("RPI", "DPCERA3M086SBEA", "INDPRO", "CUMFNS", "UNRATE", "PAYEMS",
          "CES0600000007", "CES0600000008", "WPSFD49207", "PCEPI", "HOUST",
          "S&P 500", "EXUSUKx", "GS5", "GS10", "BAAFFM",   # Carriero et al. (2024)
          "FEDFUNDS", "TB3MS", "GS1", "M2SL", "BUSLOANS", "CPIAUCSL",
          "OILPRICEx", "RETAILx", "PERMIT")                # additional series
data <- as.matrix(raw[, vars])
dates <- raw[[1]]
ok <- rowSums(is.na(data)) == 0    # keep fully observed months
data <- data[ok, , drop = FALSE]; dates <- dates[ok]
Y0 <- data[1:p, , drop = FALSE]                  # pre-sample obs (initial conditions)
Y  <- data[(p+1):nrow(data), , drop = FALSE]     # estimation sample
dates <- dates[(p+1):length(dates)]
cat(sprintf("FRED-MD: n=%d variables, sample %.2f-%.2f (%d months)\n",
            length(vars), dates[1], dates[length(dates)], length(dates)))
T <- nrow(Y)
n <- ncol(Y)
k <- n*p + 1

# prior hyperparameters
kappa1 <- 100       # prior variance on intercepts
kappa2 <- 0.2^2     # overall shrinkage on lag coefficients
rw <- 0             # zero prior mean
prior <- Minn_NCP(Y, Y0, p, kappa1, kappa2, rw)
A0 <- prior$A0; VA <- prior$VA; nu0 <- prior$nu0; S0 <- prior$S0
iva <- 1/diag(VA)               # diagonal of VA^{-1} (sparse iVA in the .m)
phi0 <- 0.95; Vphi <- 0.1^2     # common SV: phi ~ N(phi0,Vphi) on (-1,1)
nuh0 <- 20; Sh0 <- 0.05*(nuh0-1)  # sigh2 ~ IG(nuh0,Sh0)
p0a <- 10/4; p0b <- (1-1/48)*120  # outlier prob p_o ~ Beta(p0a,p0b)
ngrid <- 100                      # grid points for the U(2,10) component
o_grid <- c(1, seq(2, 10, length.out = ngrid))  # support of o_t: point mass at 1 plus grid

# construct the regressor matrix Z = [1, y_{t-1}',...,y_{t-p}']
tmpY <- rbind(Y0[(nrow(Y0)-p+1):nrow(Y0), , drop = FALSE], Y)
Z <- matrix(0, T, n*p)
for (i in 1:p) {
    Z[, ((i-1)*n+1):(i*n)] <- tmpY[(p-i+1):(nrow(tmpY)-i), , drop = FALSE]
}
Z <- cbind(1, Z)

# storage
store_A <- matrix(0, k, n)
store_Sig <- matrix(0, n, n)
store_h <- matrix(0, nsims, T)
store_o <- matrix(0, nsims, T)
store_theta <- matrix(0, nsims, 3)   # [phi, sigh2, po]
counth <- 0; countphi <- 0

# initialize the Markov chain
phi <- phi0; sigh2 <- 0.1
o <- rep(1, T); o2 <- o^2             # o2_t = o_t^2
po <- 1/48
    # initialize h with a Gaussian approximation
KA_ols <- crossprod(Z); diag(KA_ols) <- diag(KA_ols) + iva
A_ols <- solve(KA_ols, crossprod(Z, Y))
U_ols <- Y - Z %*% A_ols
C_ols <- t(chol(crossprod(U_ols)/T))              # lower factor
s2_init <- rowSums(t(forwardsolve(C_ols, t(U_ols)))^2)
h <- SV_RW_gaussian_approx(s2_init, mean(log(s2_init)), sigh2)
h <- h - mean(h)      # common SV is normalized to zero mean

cat("Starting MCMC for the VAR with common SV and outliers....\n")
t_start <- Sys.time()
for (isim in 1:(nsims+burnin)) {
    # sample Sig and A given (h, o2) with Omega^{-1} = diag(exp(-h)/o2)
    iOm <- exp(-h)/o2
    ZiOm <- t(Z*iOm)                       # k x T, = Z' * diag(iOm)
    KA <- ZiOm %*% Z
    diag(KA) <- diag(KA) + iva
    # CKA is the UPPER factor (MATLAB stores the lower one): KA = t(CKA)%*%CKA;
    # backsolve(CKA, x, transpose = TRUE) is MATLAB's CKA_lower \ x
    CKA <- chol(KA)
    Ahat <- backsolve(CKA, backsolve(CKA, iva*A0 + ZiOm %*% Y, transpose = TRUE))
    Shat <- S0 + crossprod(A0, iva*A0) + crossprod(Y, iOm*Y) -
            crossprod(Ahat, KA %*% Ahat)
    Shat <- (Shat + t(Shat))/2   # adjust for rounding errors
    Sig <- iwishrnd(Shat, nu0+T)
    CSig <- t(chol(Sig))         # lower factor
    A <- Ahat + backsolve(CKA, matrix(rnorm(k*n), k, n)) %*% t(CSig)

    # sample the common log-volatility h via the Laplace-based ARMH step
    U <- Y - Z %*% A    # reduced-form residuals
    tmp <- t(forwardsolve(CSig, t(U)))   # residuals standardized by chol(Sig)
    s2 <- rowSums(tmp^2)/o2              # per-period sum of squares over n vars
    out <- sample_CSV_h_ARMH(s2, phi, sigh2, h, n, 30)  # kappa=30 for better mixing
    h <- out$h; flag <- out$accept
    counth <- counth + flag

    # sample the outlier states o_t on a grid and the outlier probability p_o
    s2 <- rowSums(tmp^2)/exp(h)   # per-period sum of squares given h
    o_lpri <- log(c(1-po, rep(po/ngrid, ngrid)))   # prior over the grid
    for (tt in 1:T) {
        lliket <- -n*log(o_grid) - .5*s2[tt]/o_grid^2  # log-likelihood at each o
        lp <- lliket + o_lpri
        o_post <- exp(lp - max(lp))
        o_post <- o_post/sum(o_post)
        o[tt] <- o_grid[which(runif(1) < cumsum(o_post))[1]]   # inverse-CDF draw
    }
    n_out <- sum(o > 1)
    po <- rbeta(1, p0a + n_out, p0b + T - n_out)
    o2 <- o^2

    # sample sigh2
    eh <- c(h[1]*sqrt(1-phi^2), h[2:T] - phi*h[1:(T-1)])
    # R's rgamma takes shape and scale
    sigh2 <- 1/rgamma(1, shape = nuh0+T/2, scale = 1/(Sh0 + sum(eh^2)/2))

    # sample phi via an independence-chain MH step
    Kphi <- 1/Vphi + sum(h[1:(T-1)]^2)/sigh2
    phihat <- (phi0/Vphi + sum(h[1:(T-1)]*h[2:T])/sigh2)/Kphi
    phic <- phihat + rnorm(1)/sqrt(Kphi)
    gphi <- function(x) -.5*log(sigh2/(1-x^2)) - .5*(1-x^2)/sigh2*h[1]^2
    if (abs(phic) < .9999) {
        if (exp(gphi(phic)-gphi(phi)) > runif(1)) {
            phi <- phic
            countphi <- countphi + 1
        }
    }

    if (isim > burnin) {
        isave <- isim - burnin
        store_A <- store_A + A
        store_Sig <- store_Sig + Sig
        store_h[isave, ] <- h
        store_o[isave, ] <- o
        store_theta[isave, ] <- c(phi, sigh2, po)
    }

    if (isim %% 1000 == 0) {
        cat(isim, "loops... \n")
    }
}
cat(sprintf("MCMC takes %.1f seconds\n",
            as.numeric(difftime(Sys.time(), t_start, units = "secs"))))

# posterior summaries
# the level of h is not separately identified from Sig (only exp(h_t)*Sig is),
# so normalize each draw of h to zero mean before reporting the volatility
store_h <- store_h - rowMeans(store_h)
A_mean <- store_A/nsims
Sig_mean <- store_Sig/nsims
theta_mean <- colMeans(store_theta)
h_mean <- colMeans(store_h)
vol <- exp(store_h/2)  # e^{h_t/2}: normalized uncertainty measure
vol_mean <- colMeans(vol)
vol_ci <- t(apply(vol, 2, quantile, probs = c(.05, .95), type = 5))   # 90% CI
o_mean <- colMeans(store_o)
o_ci <- t(apply(store_o, 2, quantile, probs = c(.05, .95), type = 5)) # 90% CI
oprob <- colMeans(store_o > 1)                                        # P(o_t > 1)

cat(sprintf("posterior mean p_o: %.3f; months with P(o_t>1)>0.5: %d\n",
            theta_mean[3], sum(oprob > 0.5)))
cat(sprintf("acceptance rate (h): %.3f\n", counth/(nsims+burnin)))

# inefficiency factors (integrated autocorrelation times) of the T elements
# of the common log-volatility h, summarizing the mixing of the ARMH step.
# The spectral estimator uses a Bartlett window with truncation lag 500.
IF_h <- inefficiency_factor(store_h, 500)   # store_h is nsims x T
cat(sprintf("IF of h: mean %.1f, median %.1f\n", mean(IF_h), median(IF_h)))

# final results summary
cat(sprintf("phi posterior mean %.3f ; sigh2 posterior mean %.4f\n",
            theta_mean[1], theta_mean[2]))
cat(sprintf("max posterior mean volatility e^{h_t/2}: %.2f (at %.2f)\n",
            max(vol_mean), dates[which.max(vol_mean)]))

# common volatility e^{h_t/2} with 90% credible interval
plot(dates, vol_mean, type = "n", ylim = range(vol_ci),
     xlab = "", ylab = "", bty = "n", cex.axis = 1.2)
shaded_band(dates, vol_ci[, 1], vol_ci[, 2])
lines(dates, vol_mean, col = "black")
# title("common volatility e^{h_t/2}")

# outlier component o_t with 90% credible interval
plot(dates, o_mean, type = "n", ylim = range(o_ci),
     xlab = "", ylab = "", bty = "n", cex.axis = 1.2)
shaded_band(dates, o_ci[, 1], o_ci[, 2])
lines(dates, o_mean, col = "black")
# title("outlier component o_t")
