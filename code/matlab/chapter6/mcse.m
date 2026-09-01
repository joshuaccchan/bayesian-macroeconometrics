function MCSE = mcse(draws, L)
% mcse.m
% Computes Monte Carlo standard errors (MCSEs) for posterior means
% obtained from MCMC output. For each column of draws, MCSE(j) is
% sqrt(Omega_j / R), where Omega_j is the long-run variance estimated
% by the spectral variance at zero. Requires specvar0.m.
%
% Inputs:
%   draws : R-by-k matrix of MCMC draws
%   L     : truncation lag for the spectral variance estimator
%
% Output:
%   MCSE  : 1-by-k vector of Monte Carlo standard errors
[R, k] = size(draws);
MCSE = zeros(1, k);
for j = 1:k
    x = draws(:, j);
    Omega = specvar0(x, L) * R; % long-run variance of x^{(r)}
    MCSE(j) = sqrt(Omega / R);
end
end
