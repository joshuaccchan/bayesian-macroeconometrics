function logden = lmvnpdf(x,mu,Sig)
% lmvnpdf.m
% Evaluates the log density of a multivariate normal
% distribution
%
% Inputs:
% x: evaluation points, k x 1
% mu: mean vector, k x 1
% Sig: covariance matrix, k x k
%
% Output:
% logden: log density of N(mu,Sig) evaluated at x

x = x(:);
mu = mu(:);
k = length(mu);

CSig = chol(Sig,'lower');
e = CSig \ (x - mu);

logden = -0.5*k*log(2*pi) - sum(log(diag(CSig))) - 0.5*(e'*e);
end
