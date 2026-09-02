% get_SP500_data.m
%
% SP500.csv is not distributed with this repository: the S&P 500 index is
% proprietary to S&P Dow Jones Indices and cannot be redistributed. Run this
% script once to download it and write SP500.csv in the format
% SVM_SP500_HMC.m expects ([Excel serial date, index level], no header).
%
% The script reconstructs the exact sample used in the book -- daily closes
% from 2-Jan-2013 to 31-Dec-2015 (756 trading days) -- so that
% SVM_SP500_HMC.m reproduces the numbers in the book. FRED's SP500 series
% only serves the most recent 10 years and can no longer provide this window,
% so the data are pulled from Yahoo Finance's public chart API (^GSPC), whose
% closes match the book's FRED extract to the penny. The shipped DFF.csv
% covers 2-Jan-2013 to 31-Aug-2026, so the date merge covers the full sample.
%
% For a longer or more recent sample, adjust period1/period2 below (Unix
% timestamps) -- and retune the HMC step size in SVM_SP500_HMC.m: the
% published eps = 0.04 is calibrated to this T = 755 sample and yields zero
% acceptance on, e.g., the 2016-2026 window (there eps = 0.01 works).

url = ['https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC' ...
    '?period1=1356998400&period2=1452000000&interval=1d'];
opts = weboptions('UserAgent', 'Mozilla/5.0', 'Timeout', 60);
j = webread(url, opts);
res = j.chart.result;
if iscell(res), res = res{1}; end
q = res.indicators.quote;
if iscell(q), q = q{1}; end
ts = res.timestamp;                       % Unix seconds (NYSE trading days)
v  = round(q.close, 2);                   % official closes are quoted to 2 dp
dt = datetime(ts, 'ConvertFrom', 'posixtime', 'TimeZone', 'America/New_York');
dt = dateshift(dt, 'start', 'day');
dt.TimeZone = '';
d  = datenum(dt) - 693960;                % MATLAB datenum -> Excel serial date
keep = ~isnan(v) & dt >= datetime(2013,1,2) & dt <= datetime(2015,12,31);
writematrix([d(keep) v(keep)], 'SP500.csv');
fprintf('SP500.csv written: %d observations, %s to %s\n', sum(keep), ...
    datestr(dt(find(keep,1))), datestr(dt(find(keep,1,'last'))));
