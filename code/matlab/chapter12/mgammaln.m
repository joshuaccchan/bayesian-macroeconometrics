function k = mgammaln(n,x)
% mgammaln.m
% Evaluates the log of the multivariate gamma function, log Gamma_n(x).
%
% Inputs:
%   n : dimension
%   x : argument
%
% Output:
%   k : log of the multivariate gamma function evaluated at x
k = n*(n-1)/4*log(pi) + sum(gammaln((x+(0:-.5:(1-n)/2))));
end
