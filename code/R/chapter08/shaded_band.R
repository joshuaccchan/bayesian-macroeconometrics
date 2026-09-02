# shaded_band.R
# Plot a grayscale shaded credible band between pointwise lower and
# upper bounds. Useful for visualizing posterior credible intervals
# for time-varying quantities.
#
# Inputs:
#   x    : length-T vector
#   lo   : length-T vector, lower bound
#   hi   : length-T vector, upper bound
#   shade: grayscale level in [0,1] (optional, default = 0.85)
#
# The band is drawn on the current plot, so open the axes first with
# plot(..., type = "n") and add the lines afterwards.

shaded_band <- function(x, lo, hi, shade = 0.85) {
    # ensure plain vectors
    x  <- as.numeric(x)
    lo <- as.numeric(lo)
    hi <- as.numeric(hi)

    # plot shaded region
    polygon(c(x, rev(x)), c(lo, rev(hi)), col = gray(shade), border = NA)
    invisible(NULL)
}
