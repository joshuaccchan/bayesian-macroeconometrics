# VAR_NCP.R
# Forecasting with a VAR(p) under the natural conjugate (normal-inverse-
# Wishart) prior, applied to a four-variable quarterly macro panel. With the
# lag order fixed at p, the script reports the log marginal likelihood and
# runs a recursive (expanding-window) one-step-ahead forecasting exercise for
# PCE inflation from 2000Q1 to 2019Q4, reporting the RMSE and plotting the
# forecasts against the realized series. The computations are closed form, so
# the script is deterministic.
#
# Requires: Minn_NCP.R, estimate_VAR_NCP.R, ml_VAR_NCP.R, mgammaln.R, ldet.R

source("Minn_NCP.R")
source("estimate_VAR_NCP.R")
source("mgammaln.R")
source("ldet.R")
source("ml_VAR_NCP.R")

set.seed(42)   # for reproducibility
p <- 7
kappa1 <- 100; kappa2 <- 0.04; rw <- 0

# load data
data <- as.matrix(read.csv("macro4_Q.csv")[, -1])   # drop date column
Y0 <- data[1:8, ]      # initial conditions: 1960Q1-1961Q4
Y <- data[9:240, ]     # 1962Q1-2019Q4
T <- nrow(Y); n <- ncol(Y)
k <- n*p + 1

# construct prior using full sample (for marginal likelihood)
prior <- Minn_NCP(Y, Y0, p, kappa1, kappa2, rw)
post <- estimate_VAR_NCP(Y, Y0, p, prior$A0, prior$VA, prior$nu0, prior$S0)

cat(sprintf("Log ML (macro4_Q, p=%d): %.2f\n", p,
            ml_VAR_NCP(prior$VA, prior$S0, prior$nu0, post$KA, post$S_hat, T)))

# recursive forecasting from 2000Q1 to 2019Q4
# 2000Q1 corresponds to row (2000-1962)*4+1 = 153
t0 <- (2000-1962)*4 + 1
Yfull <- rbind(Y0, Y)   # stack pre-sample with sample
var_idx <- 2            # PCE inflation
nFcst <- T - t0 + 1
fcst <- numeric(nFcst)
actual <- Y[t0:T, var_idx]
fcstDates <- 2000 + 0.25*(0:(nFcst-1))

for (t in t0:T) {
    Yt <- Y[1:(t-1), ]   # estimation sample up to time t-1
    Y0t <- Y0
    # construct prior using data up to time t-1
    priort <- Minn_NCP(Yt, Y0t, p, kappa1, kappa2, rw)
    A_hat <- estimate_VAR_NCP(Yt, Y0t, p, priort$A0, priort$VA,
                              priort$nu0, priort$S0)$A_hat
    r <- nrow(Y0) + t
    # build x_t = (1, y_{t-1}', ..., y_{t-p}')
    xt <- c(1, as.vector(t(Yfull[(r-1):(r-p), , drop = FALSE])))
    yhat <- as.numeric(crossprod(xt, A_hat))
    fcst[t - t0 + 1] <- yhat[var_idx]
}
err <- actual - fcst
RMSE <- sqrt(mean(err^2))

cat(sprintf("1-step PCE inflation forecast (2000Q1-2019Q4): RMSE = %.3f\n", RMSE))

setEPS()
postscript("VAR_forecast.eps", width = 9, height = 3.5)
par(mar = c(4.2, 4.2, 1, 1))
plot(fcstDates, actual, type = "l", lty = 1, lwd = 1.8, col = "black",
     xlim = c(fcstDates[1], fcstDates[length(fcstDates)]),
     xlab = "Time", ylab = "PCE inflation", bty = "n",
     cex.lab = 1.2, cex.axis = 1.1)
lines(fcstDates, fcst, lty = 2, lwd = 1.5, col = "black")
legend("topright", legend = c("Actual", "Forecast"), lty = c(1, 2),
       lwd = c(1.8, 1.5), bty = "n", cex = 1.1)
invisible(dev.off())
