# plotCI.R
# Shades a pointwise credible band between the lower curve y1 and the upper
# curve y2 over the horizontal grid x, using the fill color C (default light
# gray). Used to draw posterior credible intervals. Adds the band to the
# current plot, so open the plot region first.
#
# Inputs:
#   x  : horizontal grid (vector)
#   y1 : lower band (vector, same length as x)
#   y2 : upper band (vector, same length as x)
#   C  : optional fill color (default light gray)

plotCI <- function(x, y1, y2, C = gray(0.85)) {
    polygon(c(x, rev(x)), c(y1, rev(y2)), col = C, border = NA)
}
