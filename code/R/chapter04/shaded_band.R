# shaded_band.R
# Plot a grayscale shaded credible band between pointwise lower and
# upper bounds. Useful for visualizing posterior credible intervals
# for time-varying quantities. Adds the band to the current plot, so
# open the plot region first (e.g. with plot(..., type = "n")).
#
# Inputs:
#   x    : length-T vector (numeric or Date)
#   lo   : length-T vector, lower bound
#   hi   : length-T vector, upper bound
#   shade: grayscale level in [0,1] (optional, default = 0.85)
#
# Output:
#   NULL (called for its side effect on the current plot)

shaded_band <- function(x, lo, hi, shade = 0.85) {

    # ensure plain vectors
    x  <- as.numeric(x)
    lo <- as.numeric(lo)
    hi <- as.numeric(hi)

    # plot shaded region
    polygon(c(x, rev(x)), c(lo, rev(hi)),
            col = gray(shade), border = NA)
    invisible(NULL)
}
