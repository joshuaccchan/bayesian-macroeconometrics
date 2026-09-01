function W = dirirnd(alpha, N)
% dirirnd.m
% Draws from a Dirichlet distribution D(alpha) using the standard
% gamma-normalization construction: if g_m ~ G(alpha_m, 1) iid, then
% (g_1, ..., g_M) / sum_m g_m has a D(alpha) distribution.
%
% Inputs:
%   alpha: M-by-1 vector of concentration parameters (alpha_m > 0)
%   N    : number of draws
%
% Output:
%   W    : N-by-M matrix; each row is an independent draw on the
%          unit simplex

alpha = alpha(:); % ensure alpha is a column vector
G = gamrnd(repmat(alpha', N, 1), 1);  % N-by-M matrix
W = G ./ sum(G, 2);
end
