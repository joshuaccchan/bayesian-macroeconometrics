function [loglik, u] = loglike_MA1(psi, e, sig2)
% loglike_MA1.m
% Gaussian log-likelihood of the linear regression model with MA(1)
% errors, evaluated on regression residuals e = y - X*beta. The MA(1)
% structure is enforced via the band matrix H_psi, and the whitened
% residuals u = H_psi^{-1} e are obtained by a banded back-solve.
%
% Inputs:
%   psi : scalar MA(1) parameter
%   e   : T-by-1 vector of regression residuals (y - X*beta)
%   sig2: innovation variance
%
% Outputs:
%   loglik: log p(e | psi, sig2) evaluated using u = H_psi^{-1} e
%   u     : implied innovations u
T = length(e);
Hpsi = spdiags([psi*ones(T,1), ones(T,1)], [-1, 0], T, T);
u = Hpsi\e;
loglik = -0.5*T*log(2*pi*sig2) - 0.5*(u'*u)/sig2;
end
