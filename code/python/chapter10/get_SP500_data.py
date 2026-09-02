# get_SP500_data.py
#
# SP500.csv is not distributed with this repository: the S&P 500 index is
# proprietary to S&P Dow Jones Indices and cannot be redistributed. Run this
# script once to download it and write SP500.csv in the format
# SVM_SP500_HMC.py expects ([Excel serial date, index level], no header).
#
# The script reconstructs the exact sample used in the book -- daily closes
# from 2-Jan-2013 to 31-Dec-2015 (756 trading days) -- so that
# SVM_SP500_HMC.py reproduces the numbers in the book. FRED's SP500 series
# only serves the most recent 10 years and can no longer provide this window,
# so the data are pulled from Yahoo Finance's public chart API (^GSPC), whose
# closes match the book's FRED extract to the penny. The shipped DFF.csv
# covers 2-Jan-2013 to 31-Aug-2026, so the date merge covers the full sample.
#
# For a longer or more recent sample, adjust period1/period2 below (Unix
# timestamps) -- and retune the HMC step size in SVM_SP500_HMC.py: the
# published eps = 0.04 is calibrated to this T = 755 sample and yields zero
# acceptance on, e.g., the 2016-2026 window (there eps = 0.01 works).

import json
import urllib.request

import numpy as np
import pandas as pd

url = ('https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC'
       '?period1=1356998400&period2=1452000000&interval=1d')
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=60) as r:
    j = json.load(r)
res = j['chart']['result'][0]
ts = np.array(res['timestamp'])           # Unix seconds (NYSE trading days)
v = np.round(np.array(res['indicators']['quote'][0]['close'], dtype=float), 2)
dates = (pd.to_datetime(ts, unit='s', utc=True)
         .tz_convert('America/New_York').normalize().tz_localize(None))
# Excel serial date: days since 30-Dec-1899 (same convention as DFF.csv)
d = (dates - pd.Timestamp('1899-12-30')).days.to_numpy()
keep = (~np.isnan(v)) & (dates >= '2013-01-02') & (dates <= '2015-12-31')
np.savetxt('SP500.csv', np.column_stack((d[keep], v[keep])),
           fmt='%.10g', delimiter=',')
t0 = pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(d[keep][0]))
t1 = pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(d[keep][-1]))
print(f'SP500.csv written: {keep.sum()} observations, '
      f'{t0:%d-%b-%Y} to {t1:%d-%b-%Y}')
