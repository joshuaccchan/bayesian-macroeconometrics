function lml = ml_VAR_NCP(VA,S0,nu0,KA,S_hat,T)
% ml_VAR_NCP.m
% Evaluates the log marginal likelihood of a VAR under the natural conjugate
% (normal-inverse-Wishart) prior, computed in logs for numerical stability.
%
% Requires: mgammaln.m, ldet.m
%
% Inputs:
%   VA    : k x k prior scale matrix, so that Cov(vec(A)|Sigma) = Sigma x VA
%   S0    : n x n prior scale matrix for the inverse-Wishart on Sigma
%   nu0   : prior degrees of freedom for the inverse-Wishart
%   KA    : k x k posterior precision matrix, KA = VA^{-1} + Z'Z
%   S_hat : n x n posterior scale matrix
%   T     : number of observations
%
% Output:
%   lml : log marginal likelihood
n = size(S0,1);
lml = -n*T/2*log(pi) - n/2*(ldet(VA) + ldet(KA)) + nu0/2*ldet(S0) ...
    - (nu0+T)/2*ldet(S_hat) + mgammaln(n,(nu0+T)/2) - mgammaln(n,nu0/2);
end
