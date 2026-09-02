# forecast_largeVAR.R
# Recursive out-of-sample forecasting exercise for the large VAR
# chapter: one-step-ahead forecasts of industrial production growth,
# CPI inflation, and the unemployment rate from a 25-variable monthly
# FRED-MD VAR with p = 12 lags, re-estimated on an expanding window at
# each origin from 2000M1 to the end of the sample.
#
# Eight model-prior combinations are compared:
#   1) homoskedastic VAR, natural conjugate Minnesota prior
#   2) common stochastic volatility, natural conjugate Minnesota prior
#   3) Cholesky stochastic volatility, independent Minnesota prior
#   4) order-invariant stochastic volatility, independent Minnesota prior
#   5) factor stochastic volatility, independent Minnesota prior
#   6) order-invariant SV, Minnesota prior with estimated tightness
#      (kappa2, kappa3 sampled)
#   7) order-invariant SV, horseshoe prior
#   8) order-invariant SV, Minnesota-type horseshoe prior (estimated
#      tightness plus local horseshoe scales)
#
# The exercise is computationally intensive: with the default settings it
# takes many hours. The MATLAB original distributes the origins over a
# parallel pool with parfor; here the loop is a plain sequential for, and
# the per-origin checkpoint files below let the run be split or resumed.
# Requires pred_largeVAR_NCP.R, pred_largeVAR_CSV.R, pred_largeVAR_SV.R,
# pred_largeVAR_OISV.R, pred_largeVAR_FSV.R, and their dependencies.

suppressMessages(library(Matrix))
source("Minn_NCP.R")
source("Minn_indep.R")
source("SVAR1.R")
source("sample_CSV_h_ARMH.R")
source("sample_SVAR1para.R")
source("sample_SVAR1para_mu.R")
source("sample_B0.R")
source("pred_largeVAR_NCP.R")
source("pred_largeVAR_CSV.R")
source("pred_largeVAR_SV.R")
source("pred_largeVAR_OISV.R")
source("pred_largeVAR_FSV.R")

p <- 12       # number of lags (monthly data)
r <- 4        # number of factors in the VAR-FSV
nsim <- 5000
burnin <- 1000
run_models <- 4:8    # subset to run, e.g., run_models <- c(1, 3)
sample_end <- 2020   # last date kept (2020.0 = 2019M12). Set to Inf for the full sample

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
# express the log-based series in percentages (100 x log-differences),
# leaving the series already in percentage points unscaled.
in_pp <- vars %in% c("CUMFNS", "UNRATE", "CES0600000007", "GS5",
                     "GS10", "BAAFFM", "FEDFUNDS", "TB3MS", "GS1")
data[, !in_pp] <- 100*data[, !in_pp]
dates <- raw[[1]]
ok <- rowSums(is.na(data)) == 0    # keep fully observed months
data <- data[ok, , drop = FALSE]; dates <- dates[ok]
pre <- dates <= sample_end + 1e-6
data <- data[pre, , drop = FALSE]; dates <- dates[pre]
Y0 <- data[1:p, , drop = FALSE]                 # pre-sample obs (initial conditions)
Y  <- data[(p+1):nrow(data), , drop = FALSE]    # estimation sample
dates <- dates[(p+1):length(dates)]
T <- nrow(Y)
n <- ncol(Y)

# forecast targets: IP, CPI inflation, unemployment rate
targets <- c(which(vars == "INDPRO"), which(vars == "CPIAUCSL"),
             which(vars == "UNRATE"))
target_names <- c("IP", "Inflation", "Unemployment")
q <- length(targets)

# prior hyperparameters
kappa1 <- 100        # intercept prior variance
kappa2 <- 0.2^2      # own-lag tightness (and overall NCP shrinkage)
kappa3 <- 0.2^2/4    # cross-lag tightness

# forecast origins: targets from 2000M1 (dates = 2000 + 1/12) to the end
# of the sample
t0 <- which(dates >= 2000 + 1/12 - 1e-6)[1]
origins <- t0:T     # forecast target is Y[t,], estimation uses Y[1:(t-1),]
nfcst <- length(origins)
fcst_dates <- dates[origins]
actual <- Y[origins, targets, drop = FALSE]

model_names <- c("Homoskedastic", "Common SV", "Cholesky SV",
                 "Order-invariant SV", "Factor SV",
                 "Minnesota (estimated)", "Horseshoe",
                 "Minnesota-type horseshoe")
nmodels <- length(model_names)

# res[i, m, ] = [pf(1:q), lpl(1:q), lpl_joint] for origin i, model m
res <- array(NA_real_, c(nfcst, nmodels, 2*q+1))

# per-origin checkpointing: each completed origin is saved to disk, so an
# interrupted run can resume where it left off, and runs with different
# run_models (e.g., split across machines) are merged automatically. Each
# (origin, model) pair is seeded independently below, so the merged
# results are identical to those from a single full run
ckdir <- file.path(getwd(), "fcst_checkpoints")
if (!dir.exists(ckdir)) dir.create(ckdir)
cktag <- Sys.getenv("COMPUTERNAME")
if (cktag == "") cktag <- "local"
cktag <- sprintf("%s_m%s", cktag, paste(run_models, collapse = ""))

load_checkpoints <- function(ckdir, ifc, nmodels, ncols) {
    # merge any saved results for origin ifc, possibly written by different
    # machines or by runs with different run_models subsets
    tmp <- matrix(NA_real_, nmodels, ncols)
    fls <- list.files(ckdir, pattern = sprintf("^origin_%03d_.*\\.rds$", ifc),
                      full.names = TRUE)
    for (fl in fls) {
        S_tmp <- readRDS(fl)$tmp
        filled <- !is.na(S_tmp)
        tmp[filled] <- S_tmp[filled]
    }
    tmp
}

save_checkpoint <- function(ckdir, ifc, tmp, cktag) {
    # save the merged results for origin ifc; the machine- and model-specific
    # file name avoids write collisions when the work is split across machines
    saveRDS(list(tmp = tmp),
            file.path(ckdir, sprintf("origin_%03d_%s.rds", ifc, cktag)))
}

progress_env <- new.env()
progress_print <- function(incr, total) {
    # running progress counter: incr = 0 resets the count and starts the
    # clock; incr = 1 (after each completed origin) prints the count,
    # elapsed time, and an estimate of the time remaining
    if (incr == 0) {
        progress_env$ndone <- 0
        progress_env$t0 <- Sys.time()
        return(invisible(NULL))
    }
    progress_env$ndone <- progress_env$ndone + incr
    elapsed <- as.numeric(difftime(Sys.time(), progress_env$t0, units = "mins"))
    ndone <- progress_env$ndone
    cat(sprintf("%d of %d origins done (elapsed %.1f min, est. remaining %.1f min)\n",
                ndone, total, elapsed, elapsed*(total - ndone)/ndone))
}

cat(sprintf("FRED-MD: n=%d variables, %d forecast origins (%.2f-%.2f)\n",
            n, nfcst, fcst_dates[1], fcst_dates[nfcst]))
cat(sprintf("models: %s\n", paste(model_names[run_models], collapse = "; ")))

progress_print(0, nfcst)   # reset the counter and start the clock

t_start <- Sys.time()
# the MATLAB original runs this loop as a parfor over a parallel pool; here
# it is a plain sequential for (a parallel back end could split the origins
# in exactly the same way, since each origin is seeded independently)
for (ifc in 1:nfcst) {
    t <- origins[ifc]
    Tt <- t - 1            # estimation sample: Y[1:Tt,]
    # load any results already saved for this origin and compute only the
    # requested models that are still missing
    tmp <- load_checkpoints(ckdir, ifc, nmodels, 2*q+1)
    todo <- run_models[apply(is.na(tmp[run_models, , drop = FALSE]), 1, any)]
    for (m in todo) {
        set.seed(1e6 + 1000*m + t)  # per-(origin, model) seed for reproducibility
        # each model is asked to forecast all n variables: the joint
        # LPL is evaluated over the full cross section, while the RMSEs
        # are computed for the three targets only
        out <- switch(as.character(m),
            "1" = pred_largeVAR_NCP(Y, Y0, Tt, p, 1:n, kappa1, kappa2, nsim),
            "2" = pred_largeVAR_CSV(Y, Y0, Tt, p, 1:n, kappa1, kappa2,
                                    nsim, burnin),
            "3" = pred_largeVAR_SV(Y, Y0, Tt, p, 1:n, "minn", kappa1,
                                   kappa2, kappa3, nsim, burnin),
            "4" = pred_largeVAR_OISV(Y, Y0, Tt, p, 1:n, "minn", kappa1,
                                     kappa2, kappa3, nsim, burnin),
            "5" = pred_largeVAR_FSV(Y, Y0, Tt, p, r, 1:n, "minn", kappa1,
                                    kappa2, kappa3, nsim, burnin),
            "6" = pred_largeVAR_OISV(Y, Y0, Tt, p, 1:n, "minnH", kappa1,
                                     kappa2, kappa3, nsim, burnin),
            "7" = pred_largeVAR_OISV(Y, Y0, Tt, p, 1:n, "hs", kappa1,
                                     kappa2, kappa3, nsim, burnin),
            "8" = pred_largeVAR_OISV(Y, Y0, Tt, p, 1:n, "mahp", kappa1,
                                     kappa2, kappa3, nsim, burnin))
        pf <- out$pf; lm_ <- out$lpl; lj <- out$lpl_joint
        tmp[m, ] <- c(pf[targets], lm_[targets], lj)
    }
    if (length(todo) > 0) {
        save_checkpoint(ckdir, ifc, tmp, cktag)
    }
    res[ifc, , ] <- tmp
    progress_print(1, nfcst)
}
cat(sprintf("total time: %.1f minutes\n",
            as.numeric(difftime(Sys.time(), t_start, units = "mins"))))

# evaluation
keep <- rep(TRUE, nfcst)
RMSE <- matrix(NA_real_, nmodels, q)
sumLPL <- rep(NA_real_, nmodels)
for (m in run_models) {
    err <- matrix(res[keep, m, 1:q], nrow = sum(keep), ncol = q) -
           actual[keep, , drop = FALSE]
    RMSE[m, ] <- sqrt(colMeans(err^2))
    sumLPL[m] <- sum(res[keep, m, 2*q+1])
}

# report the two comparison tables
cat("\nForecast performance across SV specifications (Minnesota prior)\n")
cat(sprintf("%-38s %8s %8s %8s %10s\n", "Specification",
            target_names[1], target_names[2], target_names[3], "sumLPL"))
for (m in intersect(1:5, run_models)) {
    cat(sprintf("%-38s %8.3f %8.3f %8.3f %10.1f\n", model_names[m],
                RMSE[m, 1], RMSE[m, 2], RMSE[m, 3], sumLPL[m]))
}
cat("\nForecast performance across priors (order-invariant SV)\n")
cat(sprintf("%-38s %8s %8s %8s %10s\n", "Prior",
            target_names[1], target_names[2], target_names[3], "sumLPL"))
for (m in intersect(c(4, 6, 7, 8), run_models)) {
    row_label <- if (m == 4) "Minnesota" else model_names[m]
    cat(sprintf("%-38s %8.3f %8.3f %8.3f %10.1f\n", row_label,
                RMSE[m, 1], RMSE[m, 2], RMSE[m, 3], sumLPL[m]))
}

saveRDS(list(res = res, actual = actual, fcst_dates = fcst_dates,
             RMSE = RMSE, sumLPL = sumLPL, model_names = model_names,
             target_names = target_names, targets = targets,
             p = p, r = r, nsim = nsim, burnin = burnin, kappa1 = kappa1,
             kappa2 = kappa2, kappa3 = kappa3, run_models = run_models),
        "forecast_largeVAR_results.rds")
