# shade_nber_recessions.R
# Shades NBER recession periods on the current plot. The x-axis is assumed to
# be in decimal years (e.g. 1947.00, 1947.25, ...). The shading covers the
# postwar recessions through the 2020 COVID recession.
#
# Inputs:
#   ymin, ymax : y-axis limits to cover with the recession shading
#
# In R the shading cannot be pushed behind existing lines (there is no
# uistack), so call this right after opening the plot region with
# plot(..., type = "n") and before drawing any lines.

shade_nber_recessions <- function(ymin, ymax) {

    # NBER peak-to-trough dates (month precision)
    # [peak_year peak_month  trough_year trough_month]
    R <- matrix(c(
        1948, 11, 1949, 10,
        1953,  7, 1954,  5,
        1957,  8, 1958,  4,
        1960,  4, 1961,  2,
        1969, 12, 1970, 11,
        1973, 11, 1975,  3,
        1980,  1, 1980,  7,
        1981,  7, 1982, 11,
        1990,  7, 1991,  3,
        2001,  3, 2001, 11,
        2007, 12, 2009,  6,
        2020,  2, 2020,  4), ncol = 4, byrow = TRUE)

    dec <- function(yy, mm) yy + (mm - 1)/12

    for (i in 1:nrow(R)) {
        x1 <- dec(R[i,1], R[i,2])
        x2 <- dec(R[i,3], R[i,4])

        rect(x1, ymin, x2, ymax, col = gray(0.9), border = NA)
    }
    invisible(NULL)
}
