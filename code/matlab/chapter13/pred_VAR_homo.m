function [pf, lpl] = pred_VAR_homo(Yt, Z, xt, beta0, V_Minn, nsim, burnin, var_idx, yreal)
% pred_VAR_homo.m
% Computes the one-step-ahead posterior predictive mean
% and log predictive likelihood of y_{t, var_idx} from the
% homoskedastic VAR
%     y_t = A' x_t + eps_t,   eps_t ~ N(0, Sig),
% with the independent Minnesota prior beta ~ N(beta0, diag(V_Minn))
% and the inverse-Wishart prior Sig ~ IW(nu0, S0). Two-block Gibbs
% sampler. The point forecast and LPL are aggregated within the
% function as the posterior predictive mean and the log-mean-exp of
% the per-draw Gaussian densities.
%
% Inputs:
% Yt:       T x n matrix of observations up to time t-1
% Z:        T x k matrix of regressors (intercept + p lags), k = 1+n*p
% xt:       1 x k regressor row for forecasting y_t
% beta0:    (n*k) x 1 Minnesota prior mean of beta
% V_Minn:   (n*k) x 1 diagonal entries of the Minnesota prior covariance
% nsim:     scalar; number of post-burnin MCMC draws
% burnin:   scalar; number of burnin draws
% var_idx:  scalar; index of the variable to forecast
% yreal:    scalar; realized value of y_{t, var_idx} (for LPL)
%
% Outputs:
% pf:   posterior predictive mean of y_{t, var_idx}
% lpl:  log of the predictive density at yreal, computed as the
%       log-mean-exp over MCMC draws of mixture-of-Gaussians weights

[T, n] = size(Yt);
k = size(Z, 2);
iVbeta = sparse(1:n*k, 1:n*k, 1./V_Minn);

% inverse-Wishart prior on Sig
nu0 = n + 2;  S0 = eye(n);

% precompute
ZZ = Z'*Z;  ZY = Z'*Yt;

% initialize the Gibbs sampler at OLS
A = Z \ Yt;
beta = A(:);
E = Yt - Z*A;
Sig = E'*E/T;
iSig = Sig \ speye(n);

store_mu  = zeros(nsim, 1);
store_var = zeros(nsim, 1);

for isim = 1:nsim + burnin
    % sample beta
    Kbeta = iVbeta + kron(iSig, ZZ);
    Cbeta = chol(Kbeta, 'lower');
    beta_hat = Cbeta' \ (Cbeta \ (iVbeta*beta0 + reshape(ZY*iSig, n*k, 1)));
    beta = beta_hat + Cbeta' \ randn(n*k, 1);

    % sample Sig
    A = reshape(beta, k, n);
    E = Yt - Z*A;
    Sig = iwishrnd(S0 + E'*E, nu0 + T);
    iSig = Sig \ speye(n);

    if isim > burnin
        isave = isim - burnin;
        mu_full = xt*A;
        store_mu(isave)  = mu_full(var_idx);
        store_var(isave) = Sig(var_idx, var_idx);
    end
end

% aggregate across draws
pf = mean(store_mu);
log_w = -0.5*log(2*pi*store_var) - 0.5*(yreal - store_mu).^2./store_var;
lpl = log(mean(exp(log_w - max(log_w)))) + max(log_w);
end
