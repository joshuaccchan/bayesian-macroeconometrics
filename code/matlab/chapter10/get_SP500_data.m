% get_SP500_data.m
%
% SP500.csv is not distributed with this repository: the S&P 500 index is
% proprietary to S&P Dow Jones Indices, and FRED's terms of use do not permit
% redistributing that series. Run this script once to download it from FRED and
% write SP500.csv in the format SVM_SP500_HMC.m expects ([Excel serial date,
% index level], no header).
%
% Note: FRED provides only the most recent 10 years of SP500, so a fresh
% download gives a later sample than the one used in the book (daily from
% January 2013); results will differ accordingly.

url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500';
raw = websave('SP500_raw.csv', url);
T = readtable(raw);
v = T{:,2};
if iscell(v), v = str2double(v); end     % FRED marks missing values with '.'
d = datenum(T{:,1}) - 693960;            % MATLAB datenum -> Excel serial date
keep = ~isnan(v);
writematrix([d(keep) v(keep)], 'SP500.csv');
delete(raw);
fprintf('SP500.csv written: %d observations, %s to %s\n', sum(keep), ...
    datestr(d(find(keep,1))+693960), datestr(d(find(keep,1,'last'))+693960));
