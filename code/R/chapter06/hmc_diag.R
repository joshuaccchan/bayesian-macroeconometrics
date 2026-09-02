# hmc_diag.R
# Computes MCMC diagnostics for the post burn-in HMC draws produced by
# hmc_demo.R: the inefficiency factor (integrated autocorrelation time),
# the Monte Carlo standard error of the posterior mean, and Geweke's
# convergence Z-statistic, for each coordinate of theta. Requires
# hmc_demo.R, leapfrog.R, inefficiency_factor.R, mcse.R, geweke_diag.R,
# and specvar0.R.

source("hmc_demo.R")  # obtain post burn-in draws (samples)

source("inefficiency_factor.R")
source("mcse.R")
source("geweke_diag.R")

L <- 50    # truncation lag
IF <- inefficiency_factor(samples, L)
MCSE <- mcse(samples, L)
gd <- geweke_diag(samples)
Z <- gd$Z; pval <- gd$pval

cat("\n            theta_1   theta_2\n")
cat(sprintf("IF        %8.2f  %8.2f\n", IF[1], IF[2]))
cat(sprintf("MCSE      %8.4f  %8.4f\n", MCSE[1], MCSE[2]))
cat(sprintf("Geweke Z  %8.2f  %8.2f\n", Z[1], Z[2]))
