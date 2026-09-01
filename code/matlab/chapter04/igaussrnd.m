function x = igaussrnd(psi, mu, n)
% igaussrnd.m
% Draws from the inverse Gaussian distribution IGAUSS(psi, mu) with
% kernel
%   f(x) \propto x^{-3/2} exp{ -psi (x-mu)^2 / (2 x mu^2) },  x > 0.
% Uses the standard chi-squared-based representation from
% Michael, Schucany, and Haas (1976).
%
% Inputs:
%   psi : positive shape parameter (scalar or n-by-1 vector)
%   mu  : positive mean parameter  (scalar or n-by-1 vector)
%   n   : number of draws (defaults to max(numel(psi), numel(mu)))
%
% Output:
%   x   : n-by-1 vector of draws from IGAUSS(psi, mu)

if nargin < 3
    n = max(numel(psi), numel(mu));
end

% expand scalars to length n; force column vectors
if isscalar(psi), psi = repmat(psi, n, 1); else, psi = psi(:); end
if isscalar(mu),  mu  = repmat(mu,  n, 1); else, mu  = mu(:);  end

if numel(psi) ~= n || numel(mu) ~= n
    error('psi and mu must be scalars or vectors of the same length n.');
end

% step 1: nu0 ~ \chi^2_1
nu0 = randn(n,1).^2;

% step 2: candidate draws
sqrt_term = sqrt(4.*mu.*psi.*nu0 + (mu.^2).*(nu0.^2));
x1 = mu + (mu.^2).*nu0./(2.*psi) - (mu./(2.*psi)).*sqrt_term;
x2 = (mu.^2)./x1;

% step 3: accept/reject switch
p = mu./(mu + x1);
U = rand(n,1) < p;
x = U.*x1 + (~U).*x2;
end
