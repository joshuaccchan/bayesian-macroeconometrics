# VAR_indep_IR.R
# Structural impulse response analysis of the oil market using a VAR(p) with
# an independent normal and inverse-Wishart prior. A two-block Gibbs
# sampler draws the VAR coefficients and the error covariance; at each
# post-burn-in draw the impulse responses to the three structural shocks
# (recursively identified) are computed with construct_IR.R, and the
# posterior means and pointwise 90% credible bands are plotted.
#
# Requires: construct_IR.R, plotCI.R

source("construct_IR.R")
source("plotCI.R")

set.seed(42)   # for reproducibility

p <- 24
nsim <- 10000
burnin <- 1000
n_hz <- 19     # impulse response horizon: 0 to 18 months

# load data
data <- as.matrix(read.csv("oil_SVAR_data.csv")[, 2:4])
Y_all <- data                       # full sample: 1973M2-2019M12
Y0 <- Y_all[1:p, ]                  # initial conditions
Y <- Y_all[(p+1):nrow(Y_all), ]     # estimation sample
T <- nrow(Y); n <- ncol(Y)
k <- 1 + n*p     # number of coefficients per equation

# prior hyperparameters
# prior mean: variable 1 (growth rate) -> 0;
# variables 2,3 (levels) -> random walk
beta0 <- numeric(n*k)
for (j in 2:n) {
    beta0[(j-1)*k + 1 + j] <- 1   # first own lag = 1 for level variables
}
iVbeta <- diag(n*k)/100   # n*k = 219 here, so a dense identity is cheap
nu0 <- n + 2;  S0 <- diag(n)

# construct the T x k regressor matrix Z
tmpY <- rbind(Y0[(nrow(Y0)-p+1):nrow(Y0), ], Y)
Z <- matrix(0, T, n*p)
for (i in 1:p) {
    Z[, ((i-1)*n+1):(i*n)] <- tmpY[(p-i+1):(nrow(tmpY)-i), ]
}
Z <- cbind(1, Z)
ZZ <- crossprod(Z); ZY <- crossprod(Z, Y)

# initialize the Gibbs sampler at OLS
A <- qr.solve(Z, Y)
beta <- as.vector(A)
E <- Y - Z %*% A
Sig <- crossprod(E)/T
iSig <- solve(Sig)

# storage for impulse responses: nsim x n_hz x n variables x n shocks
store_yIR <- array(0, c(nsim, n_hz, n, n))

for (isim in 1:(nsim + burnin)) {
    # sample beta
    Kbeta <- iVbeta + kronecker(iSig, ZZ)
    Cbeta <- chol(Kbeta)   # UPPER factor, Kbeta = t(Cbeta) %*% Cbeta
    beta_hat <- backsolve(Cbeta, forwardsolve(t(Cbeta),
                          as.numeric(iVbeta %*% beta0 + as.vector(ZY %*% iSig))))
    beta <- as.numeric(beta_hat + backsolve(Cbeta, rnorm(n*k)))

    # sample Sigma from IW(S0 + E'E, nu0 + T): if Zw is a (nu0+T) x n standard
    # normal matrix and C is the lower Cholesky factor of S0 + E'E, then
    # C*(Zw'Zw)^{-1}*C' ~ IW(S0 + E'E, nu0 + T)
    E <- Y - Z %*% matrix(beta, k, n)
    CS <- t(chol(S0 + crossprod(E)))
    Zw <- matrix(rnorm((nu0 + T)*n), nu0 + T, n)
    Sig <- CS %*% solve(crossprod(Zw), t(CS))
    Sig <- (Sig + t(Sig))/2
    iSig <- solve(Sig)

    # compute impulse responses
    if (isim > burnin) {
        isave <- isim - burnin
        for (jj in 1:n) {
            # normalize: each shock raises the oil price
            # shock 1 (supply): negative supply shock -> raise price
            # shocks 2,3 (demand): positive demand shock -> raise price
            shock <- numeric(n)
            if (jj == 1) {
                shock[jj] <- -1   # negative oil supply shock
            } else {
                shock[jj] <- 1    # positive demand shock
            }
            yIR <- construct_IR(beta, Sig, n_hz, shock)
            store_yIR[isave, , , jj] <- yIR
        }
    }
}

# cumulate oil production responses (variable 1 is a growth rate)
for (jj in 1:n) {
    store_yIR[, , 1, jj] <- t(apply(store_yIR[, , 1, jj], 1, cumsum))
}

# posterior mean and 90% credible intervals
yIR_mean <- apply(store_yIR, c(2, 3, 4), mean)
yIR_lo <- apply(store_yIR, c(2, 3, 4),
                function(v) quantile(v, .05, type = 5, names = FALSE))
yIR_hi <- apply(store_yIR, c(2, 3, 4),
                function(v) quantile(v, .95, type = 5, names = FALSE))

# plot: 3 x 3 grid (rows = variables, columns = shocks)
varnames <- c("Oil production", "Real activity", "Real price of oil")
shocknames <- c("Oil supply shock", "Aggregate demand shock",
                "Oil-specific demand shock")
hz <- 0:(n_hz-1)

# print the cumulated impact-plus-18-month responses of the real oil price
cat("Response of the real price of oil at horizon 18 (post. mean [90% CI]):\n")
for (jj in 1:n) {
    cat(sprintf("  %s: %.2f [%.2f, %.2f]\n", shocknames[jj],
                yIR_mean[n_hz, 3, jj], yIR_lo[n_hz, 3, jj],
                yIR_hi[n_hz, 3, jj]))
}

setEPS()
postscript("oil_SVAR_IR.eps", width = 8, height = 4)
par(mfrow = c(n, n), mar = c(3.2, 3.6, 2, 0.6), mgp = c(2.2, 0.7, 0))
for (ii in 1:n) {       # response variable (row)
    for (jj in 1:n) {   # shock (column)
        yl <- range(c(yIR_lo[, ii, jj], yIR_hi[, ii, jj]))
        yl <- c(min(yl[1], -0.5), max(yl[2], 0.5))
        plot(hz, yIR_mean[, ii, jj], type = "n", xlim = c(-0.5, n_hz-1),
             ylim = yl, bty = "n", xlab = "", ylab = "", cex.axis = 0.9,
             main = if (ii == 1) shocknames[jj] else "", cex.main = 1)
        plotCI(hz, yIR_lo[, ii, jj], yIR_hi[, ii, jj])
        lines(hz, yIR_mean[, ii, jj], col = "black", lwd = 1.5)
        abline(h = 0, col = "black", lwd = 0.5)
        if (jj == 1) title(ylab = varnames[ii], cex.lab = 0.95)
        if (ii == n) title(xlab = "Months", cex.lab = 0.95)
    }
}
invisible(dev.off())
