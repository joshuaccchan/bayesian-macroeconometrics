function [pf, lpl_joint, lpl] = pred_largeVAR_NCP(Y, Y0, Tt, p, ...
    targets, kappa1, kappa2, nsim)
% This function computes the one-step-ahead posterior predictive means
% and log predictive likelihoods of the target variables from the
% homoskedastic VAR
%     y_t = A' x_t + eps_t,   eps_t ~ N(0, Sig),
% with the natural conjugate prior (A, Sig) ~ NIW(A0, VA, nu0, S0)
% elicited Minnesota-style by Minn_NCP.m. The NIW posterior is
% available in closed form, so each iteration draws (A, Sig) directly
% from the posterior and evaluates the predictive density; the log
% predictive likelihoods are then formed by averaging over the draws.
%
% Inputs:
% Y:        T x n matrix of observations
% Y0:       p0 x n matrix of pre-sample observations (p0 >= p)
% Tt:       scalar; forecast origin (estimation uses Y(1:Tt,:), the
%           forecast target is Y(Tt+1,:))
% p:        scalar; lag order
% targets:  q x 1 vector of indices of the target variables
% kappa1:   scalar; prior variance on intercepts
% kappa2:   scalar; overall shrinkage on lag coefficients
% nsim:     scalar; number of posterior draws
%
% Outputs:
% pf:        1 x q posterior predictive means of the targets
% lpl_joint: scalar; log of the joint predictive density of the q
%            targets at their realized values (log-mean-exp over draws)
% lpl:       1 x q marginal log predictive likelihoods of the targets

[~, n] = size(Y);
k = 1 + n*p;
q = length(targets);
Yt = Y(1:Tt, :);
yreal = Y(Tt+1, targets);

% construct the regressor matrix Z = [1, y_{t-1}', ..., y_{t-p}']
tmpY = [Y0(end-p+1:end, :); Yt];
Z = zeros(Tt, n*p);
for i = 1:p
    Z(:, (i-1)*n+1:i*n) = tmpY(p-i+1:end-i, :);
end
Z = [ones(Tt, 1) Z];

% natural conjugate prior and analytical posterior
[A0, VA, nu0, S0] = Minn_NCP(Yt, Y0, p, kappa1, kappa2, 0);
iVA = sparse(1:k, 1:k, 1./diag(VA));
KA = iVA + Z'*Z;
CKA = chol(KA, 'lower');
A_hat = CKA'\(CKA\(iVA*A0 + Z'*Yt));
S_hat = S0 + A0'*iVA*A0 + Yt'*Yt - A_hat'*KA*A_hat;
S_hat = (S_hat + S_hat')/2;   % adjust for rounding errors
nu_hat = nu0 + Tt;

% regressor row for forecasting Y(Tt+1,:)
xtp1 = [1, reshape(Yt(end:-1:end-p+1, :)', 1, n*p)];

store_lden = zeros(nsim, q+1);
store_mu = zeros(nsim, q);
for isim = 1:nsim
    % sample Sig and A from the NIW posterior
    Sig = iwishrnd(S_hat, nu_hat);
    CSig = chol(Sig, 'lower');
    A = A_hat + (CKA'\randn(k, n))*CSig';

    % evaluate the predictive density at the realized targets
    mu_full = xtp1*A;
    mu_q = mu_full(targets);
    Sig_q = Sig(targets, targets);
    CSig_q = chol(Sig_q, 'lower');
    u = CSig_q\(yreal - mu_q)';
    lden_joint = -q/2*log(2*pi) - sum(log(diag(CSig_q))) - 0.5*(u'*u);
    dSig_q = diag(Sig_q)';
    lden = -0.5*log(2*pi*dSig_q) - 0.5*(yreal - mu_q).^2./dSig_q;
    store_mu(isim, :) = mu_q;
    store_lden(isim, :) = [lden lden_joint];
end

% aggregate across draws
pf = mean(store_mu);
tmpmax = max(store_lden);
lpls = log(mean(exp(store_lden - tmpmax))) + tmpmax;
lpl = lpls(1:q);
lpl_joint = lpls(q+1);
end
