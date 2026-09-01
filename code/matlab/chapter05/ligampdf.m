function logden = ligampdf(x,a,b)
% ligampdf.m
% Evaluates the log density of the inverse-gamma distribution.
%
% Parameterization: p(x) = b^a / Gamma(a) * x^(-(a+1)) * exp(-b/x)
%
% Inputs:
% x: evaluation point; must be positive
% a: shape parameter; must be positive
% b: scale parameter; must be positive
% The inputs x, a, and b may be scalars or arrays of compatible sizes.
%
% Output:
% logden: log density of IG(a,b) evaluated at x
logden = a.*log(b) - gammaln(a) - (a+1).*log(x) - b./x;
end
