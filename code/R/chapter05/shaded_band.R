# shaded_band.R
# Plot a grayscale shaded credible band between pointwise lower and
# upper bounds. Useful for visualizing posterior credible intervals
# for time-varying quantities. Adds the band to the current plot.
#
# Inputs:
#   x    : T-vector (numeric or Date)
#   lo   : T-vector, lower bound
#   hi   : T-vector, upper bound
#   shade: grayscale level in [0,1] (optional, default = 0.85)

shaded_band <- function(x, lo, hi, shade = 0.85) {
    # ensure plain numeric vectors (Date axes are numeric internally)
    x <- as.numeric(x)
    lo <- as.numeric(lo)
    hi <- as.numeric(hi)

    # plot shaded region
    polygon(c(x, rev(x)), c(lo, rev(hi)),
            col = gray(shade), border = NA)
}
