function [mu, s2f, s2y] = gpr_predict(y, X, Xstar, hp)
% gpr_predict.m
% Posterior predictive moments for Gaussian process regression with an ARD
% squared exponential kernel, at test inputs Xstar given hyperparameters hp:
%   E[f(x_{T+1}) | y]   = k_{T+1}' * Ksig^{-1} * y,
%   Var[f(x_{T+1}) | y] = k_{T+1,T+1} - k_{T+1}' * Ksig^{-1} * k_{T+1},
% where Ksig = K + sig2*I and k_{T+1} is the cross-covariance between a test
% input and the T training inputs. Everything goes through the Cholesky factor
% of Ksig.
%
% Inputs:
%   y     : T-by-1 response
%   X     : T-by-d training inputs
%   Xstar : m-by-d test inputs
%   hp    : struct with fields sigf2 (scalar), lam (1-by-d), sig2 (scalar)
%
% Outputs:
%   mu  : m-by-1 posterior mean of f at Xstar
%   s2f : m-by-1 posterior variance of f at Xstar
%   s2y : m-by-1 predictive variance of a noisy observation (= s2f + sig2)

[T,d] = size(X);
m     = size(Xstar,1);
sigf2 = hp.sigf2;
lam   = hp.lam(:)';
sig2  = hp.sig2;

% ARD scaled squared distances: training-training (M) and training-test (Ms)
M  = zeros(T);
Ms = zeros(T,m);
for j = 1:d
    xj = X(:,j);
    M  = M  + ((xj - xj').^2)/(lam(j)^2);
    Ms = Ms + ((xj - Xstar(:,j)').^2)/(lam(j)^2);
end
Ksig = sigf2*exp(-0.5*M) + sig2*eye(T);  % Ksig = K + sig2 I
Ks   = sigf2*exp(-0.5*Ms);               % cross-covariances k_{T+1}
C    = chol(Ksig,'lower');

alpha = C'\(C\y);  % Ksig^{-1} y
mu    = Ks'*alpha; % posterior mean  k_{T+1}' Ksig^{-1} y

v   = C\Ks;
s2f = max(sigf2 - sum(v.^2,1)', 0);  % k_{T+1,T+1} - k_{T+1}' Ksig^{-1} k_{T+1}
s2y = s2f + sig2;                    % add noise variance for y
end
