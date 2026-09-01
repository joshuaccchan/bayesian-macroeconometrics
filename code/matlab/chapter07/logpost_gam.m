function logden = logpost_gam(gam,y,X,sig2,bp,iVbeta)
% logpost_gam.m
% Evaluates the log of the collapsed posterior kernel p(gam | y, sig2)
% for the SSVS regression, with the regression coefficients beta
% integrated out analytically. The returned value sums the marginal
% log-likelihood log p(y | gam, sig2) and the log-prior log p(gam)
% under independent Bernoulli(bp) inclusion priors. If the implied
% precision matrix is not positive definite, returns -Inf.
%
% Inputs:
%   gam   : k-by-1 vector of inclusion indicators in {0,1}
%   y     : T-by-1 response
%   X     : T-by-k design matrix
%   sig2  : scalar error variance
%   bp    : k-by-1 vector of prior inclusion probabilities
%   iVbeta: k-by-k prior precision of beta
%
% Output:
%   logden: log p(gam | y, sig2) up to a normalising constant
k = size(gam,1);
Gam = sparse(1:k,1:k,gam);
Xtilde = X * Gam;

iDbeta = iVbeta + (Xtilde'*Xtilde)/sig2;
[C,flag] = chol(iDbeta);
if flag > 0
    logden = -Inf;
    return
end

beta_hat = iDbeta \ (Xtilde'*y/sig2);
logden = -sum(log(diag(C))) + 0.5*beta_hat'*iDbeta*beta_hat ...
       + sum(gam(2:end).*log(bp(2:end)) + (1-gam(2:end)).*log(1-bp(2:end)));
end