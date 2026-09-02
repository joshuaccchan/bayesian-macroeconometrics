# VAR_NCP_lag.R
# Selects the lag order of the VAR under the natural conjugate prior by
# computing the log marginal likelihood for p = 1,...,8 on macro4_Q, holding
# the estimation sample fixed across lag lengths. The computations are
# closed form, so the script is deterministic.
#
# Requires: Minn_NCP.R, estimate_VAR_NCP.R, ml_VAR_NCP.R, mgammaln.R, ldet.R

source("Minn_NCP.R")
source("estimate_VAR_NCP.R")
source("mgammaln.R")
source("ldet.R")
source("ml_VAR_NCP.R")

kappa1 <- 100; kappa2 <- 0.04; rw <- 0

# load data
data <- as.matrix(read.csv("macro4_Q.csv")[, -1])   # drop date column
Y0 <- data[1:8, ]      # initial conditions: 1960Q1-1961Q4
Y <- data[9:240, ]     # 1962Q1-2019Q4
T <- nrow(Y)

# log marginal likelihood for each lag length
lml <- numeric(8)
for (p in 1:8) {
    prior <- Minn_NCP(Y, Y0, p, kappa1, kappa2, rw)
    post <- estimate_VAR_NCP(Y, Y0, p, prior$A0, prior$VA, prior$nu0, prior$S0)
    lml[p] <- ml_VAR_NCP(prior$VA, prior$S0, prior$nu0, post$KA, post$S_hat, T)
    cat(sprintf("log ML (p = %d): %.1f\n", p, lml[p]))
}
