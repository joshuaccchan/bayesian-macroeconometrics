# shaded_band.R
# Plot a grayscale shaded credible band between pointwise lower and
# upper bounds. Useful for visualizing posterior credible intervals
# for time-varying quantities.
#
# Inputs:
#   x    : length-T vector (numeric or Date)
#   lo   : length-T vector, lower bound
#   hi   : length-T vector, upper bound
#   shade: grayscale level in [0,1] (optional, default = 0.85)
#
# Output:
#   NULL (the band is added to the current plot)
#
# Unlike MATLAB's fill, R's polygon draws on an existing plot region, so
# set up the axes first (e.g. plot(..., type = "n")) and call this before
# the lines that go on top of the band.

shaded_band <- function(x, lo, hi, shade = 0.85) {
    # ensure plain numeric vectors (Dates are drawn on their numeric scale)
    x  <- as.numeric(x)
    lo <- as.numeric(lo)
    hi <- as.numeric(hi)

    # plot shaded region
    polygon(c(x, rev(x)), c(lo, rev(hi)),
            col = gray(shade), border = NA)
}
