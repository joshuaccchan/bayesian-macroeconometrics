function h = shaded_band(x, lo, hi, shade)
% shaded_band.m
% Plot a grayscale shaded credible band between pointwise lower and
% upper bounds. Useful for visualizing posterior credible intervals
% for time-varying quantities.
%
% Inputs:
%   x    : T-by-1 vector (numeric or datetime)
%   lo   : T-by-1 vector, lower bound
%   hi   : T-by-1 vector, upper bound
%   shade: grayscale level in [0,1] (optional, default = 0.85)
%
% Output:
%   h    : handle to the fill object

if nargin < 4
    shade = 0.85;
end

% ensure column vectors
x  = x(:);
lo = lo(:);
hi = hi(:);

% plot shaded region
h = fill([x; flipud(x)], [lo; flipud(hi)], ...
         shade*[1 1 1], 'EdgeColor','none');
end
