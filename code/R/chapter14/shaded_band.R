# shaded_band.R
# Plot a grayscale shaded credible band between pointwise lower and
# upper bounds. Useful for visualizing posterior credible intervals
# for time-varying quantities.
#
# Unlike MATLAB's fill, R's polygon draws into an existing plot, so the
# caller opens the plot region first (typically with type = "n") and then
# adds the band and the posterior mean line.
#
# Inputs:
#   x    : length-T vector
#   lo   : length-T vector, lower bound
#   hi   : length-T vector, upper bound
#   shade: grayscale level in [0,1] (optional, default = 0.85)

shaded_band <- function(x, lo, hi, shade = 0.85) {
    x  <- as.numeric(x)
    lo <- as.numeric(lo)
    hi <- as.numeric(hi)

    # plot shaded region
    polygon(c(x, rev(x)), c(lo, rev(hi)), col = gray(shade), border = NA)
}
