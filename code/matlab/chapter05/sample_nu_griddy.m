function nu = sample_nu_griddy(lam, nu_ub, n_grid)
% sample_nu_griddy.m
% Griddy-Gibbs update for the degrees-of-freedom parameter nu in a
% Student-t regression model with latent scale variables lambda. The
% conditional density p(nu | lambda) is supported on (2, nu_ub).
%
% Inputs:
%   lam   : T-by-1 vector of latent scale variables
%   nu_ub : upper bound for nu
%   n_grid: number of grid points
%
% Outputs:
%   nu    : draw from p(nu | lambda)

T = length(lam);
sum_loglam = sum(log(lam));
sum_ilam = sum(1./lam);

% log kernel of p(nu | lambda)
log_kernel = @(x) T*((x/2).*log(x/2) - gammaln(x/2)) ...
    - (x/2 + 1) * sum_loglam - (x/2) * sum_ilam;

nu = griddy_gibbs(log_kernel, 2, nu_ub, n_grid);
end
