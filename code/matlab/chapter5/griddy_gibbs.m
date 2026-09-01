function [x_draw, x_grid, logf_grid] = ...
    griddy_gibbs(logf, a, b, n_grid)
% griddy_gibbs.m
% Generic Griddy-Gibbs sampler for a univariate target density with
% bounded support (a,b), given an unnormalized log-density logf(x).
%
% Inputs:
%   logf  : function handle returning log unnormalized density at x;
%           must accept a vector input and return a vector output of
%           the same size
%   a, b  : lower and upper bounds (a < b), finite
%   n_grid: number of grid points (integer >= 2)
%
% Outputs:
%   x_draw   : one draw from the Griddy-Gibbs approximation to the target
%   x_grid   : grid points used
%   logf_grid: logf evaluated on x_grid

% jittered grid to avoid repeated draws
step = (b - a) / (n_grid + 1); % grid spacing
x_grid = a + step*(1:n_grid)';    
x_grid = x_grid + (rand(n_grid,1) - 0.5)*step; % small jitter
    % enforce strict bounds
x_grid = min(max(x_grid, a + eps), b - eps);

% evaluate logf on the grid and normalize
logf_grid = logf(x_grid);
w = exp(logf_grid - max(logf_grid));
w = w / sum(w); % normalize to sum to 1

cdf_grid = cumsum(w);
u = rand;
idx = find(cdf_grid >= u, 1, 'first');
x_draw = x_grid(idx);
end
