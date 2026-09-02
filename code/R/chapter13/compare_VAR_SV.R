# compare_VAR_SV.R
# Recursive one-step-ahead forecasting of PCE inflation across three VAR
# specifications: homoskedastic, Cholesky stochastic volatility, and
# order-invariant stochastic volatility.
#
# Requires: Minn_indep.R, pred_VAR_homo.R, pred_VAR_SV.R, pred_VAR_OISV.R,
#           SURform2.R, SVRW.R, sample_B0.R, SVAR1.R, sample_SVAR1para.R

suppressMessages(library(Matrix))
source("Minn_indep.R")
source("pred_VAR_homo.R")
source("pred_VAR_SV.R")
source("pred_VAR_OISV.R")

set.seed(42)   # for reproducibility
p <- 7
nsim <- 10000
burnin <- 1000
kappa1 <- 100         # intercept prior variance
kappa2 <- 0.2^2       # own-lag tightness
kappa3 <- 0.2^2/4     # cross-lag tightness
rw      <- 0          # 0 = zero prior mean (growth-rate data); 1 = RW
var_idx <- 2          # PCE inflation

# load data
data <- as.matrix(read.csv("macro4_Q.csv")[, -1])   # drop date column
Y0 <- data[1:8, ]        # initial conditions: 1960Q1-1961Q4
Y  <- data[9:260, ]      # 1962Q1-2024Q4
T <- nrow(Y)
n <- ncol(Y)
k <- 1 + n*p

# recursive forecasting from 2000Q1 to 2024Q4
t0     <- (2000 - 1962)*4 + 1
Yfull  <- rbind(Y0, Y)
nFcst  <- T - t0 + 1
fcst_homo <- numeric(nFcst)
LPL_homo  <- numeric(nFcst)
fcst_SV   <- numeric(nFcst)
LPL_SV    <- numeric(nFcst)
fcst_OISV <- numeric(nFcst)
LPL_OISV  <- numeric(nFcst)
actual    <- Y[t0:T, var_idx]
fcstDates <- 2000 + 0.25*(0:(nFcst-1))

cat("Recursive one-step-ahead forecasts ...\n")
t_start <- Sys.time()
for (t in t0:T) {
    Yt <- Y[1:(t-1), , drop = FALSE]
    Tt <- nrow(Yt)
    yreal <- Y[t, var_idx]

    # independent Minnesota prior on beta (data up to t-1)
    Minn <- Minn_indep(p, kappa1, kappa2, kappa3, Y0, Yt, rw)
    beta0  <- Minn$beta_Minn
    V_Minn <- Minn$V_Minn

    # regressor matrix Z shared across models
    tmpY <- rbind(Y0[(nrow(Y0)-p+1):nrow(Y0), , drop = FALSE], Yt)
    Z <- matrix(1, Tt, k)
    for (i in 1:p) {
        Z[, (2 + (i-1)*n):(1 + i*n)] <- tmpY[(p-i+1):(nrow(tmpY)-i), ]
    }

    # regressor row x_t for forecasting Y[t, ]
    r  <- nrow(Y0) + t
    xt <- c(1, as.vector(t(Yfull[(r-1):(r-p), , drop = FALSE])))

    # homoskedastic VAR
    res_homo <- pred_VAR_homo(Yt, Z, xt, beta0, V_Minn, nsim, burnin,
                              var_idx, yreal)
    fcst_homo[t-t0+1] <- res_homo$pf
    LPL_homo[t-t0+1]  <- res_homo$lpl

    # VAR with Cholesky stochastic volatility
    res_SV <- pred_VAR_SV(Yt, Z, xt, beta0, V_Minn, nsim, burnin,
                          var_idx, yreal)
    fcst_SV[t-t0+1] <- res_SV$pf
    LPL_SV[t-t0+1]  <- res_SV$lpl

    # VAR with order-invariant stochastic volatility
    res_OISV <- pred_VAR_OISV(Yt, Z, xt, beta0, V_Minn, nsim, burnin,
                              var_idx, yreal)
    fcst_OISV[t-t0+1] <- res_OISV$pf
    LPL_OISV[t-t0+1]  <- res_OISV$lpl

    if ((t - t0 + 1) %% 10 == 0) {
        cat(sprintf("  forecast %d / %d done\n", t - t0 + 1, nFcst))
        flush.console()
    }
}
cat(sprintf("Elapsed: %.1f sec\n",
            as.numeric(difftime(Sys.time(), t_start, units = "secs"))))

err_homo    <- actual - fcst_homo
RMSE_homo   <- sqrt(mean(err_homo^2))
sumLPL_homo <- sum(LPL_homo)
cat(sprintf("Homoskedastic VAR  : RMSE = %.4f   sum LPL = %.4f\n",
            RMSE_homo, sumLPL_homo))

err_SV    <- actual - fcst_SV
RMSE_SV   <- sqrt(mean(err_SV^2))
sumLPL_SV <- sum(LPL_SV)
cat(sprintf("Cholesky SV VAR    : RMSE = %.4f   sum LPL = %.4f\n",
            RMSE_SV, sumLPL_SV))

err_OISV    <- actual - fcst_OISV
RMSE_OISV   <- sqrt(mean(err_OISV^2))
sumLPL_OISV <- sum(LPL_OISV)
cat(sprintf("Order-invariant SV : RMSE = %.4f   sum LPL = %.4f\n",
            RMSE_OISV, sumLPL_OISV))
