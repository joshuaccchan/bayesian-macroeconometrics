function beta_hat = fit_BayesRidge(y,X,lambda)
% fit_BayesRidge.m
% Returns the posterior-mean ridge estimator under an isotropic
% Gaussian prior beta_j ~ N(0, 1/lambda) and known unit error
% variance. The estimator is beta_hat = (lambda*I + X'X)^{-1} X'y.
%
% Inputs:
%   y     : T-by-1 response
%   X     : T-by-k design matrix
%   lambda: ridge regularisation parameter (prior precision scale)
%
% Output:
%   beta_hat: k-by-1 ridge posterior mean
k = size(X,2);
XX = X'*X; Xy = X'*y;
Dbeta = (lambda*speye(k) + XX)\speye(k);
beta_hat = Dbeta*Xy;
end