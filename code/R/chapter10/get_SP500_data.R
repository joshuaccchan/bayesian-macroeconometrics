# get_SP500_data.R
#
# SP500.csv is not distributed with this repository: the S&P 500 index is
# proprietary to S&P Dow Jones Indices and cannot be redistributed. Run this
# script once to download it and write SP500.csv in the format
# SVM_SP500_HMC.R expects ([Excel serial date, index level], no header).
#
# The script reconstructs the exact sample used in the book -- daily closes
# from 2-Jan-2013 to 31-Dec-2015 (756 trading days) -- so that
# SVM_SP500_HMC.R reproduces the numbers in the book. FRED's SP500 series
# only serves the most recent 10 years and can no longer provide this window,
# so the data are pulled from Yahoo Finance's public chart API (^GSPC), whose
# closes match the book's FRED extract to the penny. The shipped DFF.csv
# covers 2-Jan-2013 to 31-Aug-2026, so the date merge covers the full sample.
#
# For a longer or more recent sample, adjust period1/period2 below (Unix
# timestamps) -- and retune the HMC step size in SVM_SP500_HMC.R: the
# published eps = 0.04 is calibrated to this T = 755 sample and yields zero
# acceptance on, e.g., the 2016-2026 window (there eps = 0.01 works).
#
# Base R has no JSON parser, so the two arrays needed ("timestamp" and the
# quote "close") are pulled out of the fixed-shape response with a regular
# expression. Note that "close" is matched with its opening quote, which keeps
# it from matching the "adjclose" array.

url_str <- paste0("https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC",
                  "?period1=1356998400&period2=1452000000&interval=1d")
options(timeout = 60)
con <- url(url_str, open = "rb", method = "libcurl",
           headers = c("User-Agent" = "Mozilla/5.0"))
txt <- paste(readLines(con, warn = FALSE), collapse = "")
close(con)

get_array <- function(txt, key) {
    m <- regmatches(txt, regexpr(paste0('"', key, '":\\[[^]]*\\]'), txt))
    if (length(m) == 0)
        stop(sprintf("get_SP500_data: array \"%s\" not found in the response.", key))
    m <- sub(paste0('"', key, '":\\['), "", m)
    m <- sub("\\]$", "", m)
    suppressWarnings(as.numeric(strsplit(m, ",")[[1]]))   # "null" -> NA
}

ts <- get_array(txt, "timestamp")   # Unix seconds (NYSE trading days)
v  <- round(get_array(txt, "close"), 2)   # official closes are quoted to 2 dp
dt <- as.POSIXct(ts, origin = "1970-01-01", tz = "UTC")
attr(dt, "tzone") <- "America/New_York"
dt <- as.Date(format(dt, "%Y-%m-%d"))   # start of the New York trading day
d  <- as.numeric(dt - as.Date("1899-12-30"))   # Excel serial date
keep <- !is.na(v) & dt >= as.Date("2013-01-02") & dt <= as.Date("2015-12-31")
write.table(cbind(d[keep], v[keep]), "SP500.csv", sep = ",",
            row.names = FALSE, col.names = FALSE)
cat(sprintf("SP500.csv written: %d observations, %s to %s\n", sum(keep),
            format(dt[keep][1], "%d-%b-%Y"),
            format(dt[keep][sum(keep)], "%d-%b-%Y")))
