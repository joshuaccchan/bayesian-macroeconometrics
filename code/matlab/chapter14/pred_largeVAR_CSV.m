function [pf, lpl_joint, lpl] = pred_largeVAR_CSV(Y, Y0, Tt, p, ...
    targets, kappa1, kappa2, nsim, burnin)
% This function computes the one-step-ahead posterior predictive means
% and log predictive likelihoods of the target variables from the VAR
% with common stochastic volatility (Section 14.2),
%     y_t = A' x_t + eps_t,   eps_t ~ N(0, exp(h_t)*Sig),
%     h_t = phi*h_{t-1} + u_t^h,   u_t^h ~ N(0, sigh2),
%     h_1 ~ N(0, sigh2/(1 - phi^2)),
% with the natural conjugate prior (A, Sig) ~ NIW(A0, VA, nu0, S0)
% elicited Minnesota-style by Minn_NCP.m.
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
% nsim:     scalar; number of post-burnin MCMC draws
% burnin:   scalar; number of burnin draws
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

% priors for the volatility parameters
phi0 = 0.98;  Vphi = 0.05^2;     % phi ~ TN_(-1,1)(phi0, Vphi)
nu_h = 3;  S_h = 0.1*(nu_h - 1); % sigh2 ~ IG(nu_h, S_h)

% construct the regressor matrix Z = [1, y_{t-1}', ..., y_{t-p}']
tmpY = [Y0(end-p+1:end, :); Yt];
Z = zeros(Tt, n*p);
for i = 1:p
    Z(:, (i-1)*n+1:i*n) = tmpY(p-i+1:end-i, :);
end
Z = [ones(Tt, 1) Z];

% natural conjugate prior
[A0, VA, nu0, S0] = Minn_NCP(Yt, Y0, p, kappa1, kappa2, 0);
iVA = sparse(1:k, 1:k, 1./diag(VA));

% regressor row for forecasting Y(Tt+1,:)
xtp1 = [1, reshape(Yt(end:-1:end-p+1, :)', 1, n*p)];

% initialize the Markov chain
phi = phi0;
sigh2 = 0.05;
h = zeros(Tt, 1);

store_lden = zeros(nsim, q+1);
store_mu = zeros(nsim, q);
for isim = 1:nsim + burnin
    % sample Sig and A given h with Omega^{-1} = diag(exp(-h))
    iOm = sparse(1:Tt, 1:Tt, exp(-h));
    ZiOm = Z'*iOm;
    KA = iVA + ZiOm*Z;
    CKA = chol(KA, 'lower');
    A_hat = CKA'\(CKA\(iVA*A0 + ZiOm*Yt));
    S_hat = S0 + A0'*iVA*A0 + Yt'*iOm*Yt - A_hat'*KA*A_hat;
    S_hat = (S_hat + S_hat')/2;   % adjust for rounding errors
    Sig = iwishrnd(S_hat, nu0 + Tt);
    CSig = chol(Sig, 'lower');
    A = A_hat + (CKA'\randn(k, n))*CSig';

    % sample the common log-volatility h via the Laplace-based ARMH step
    U = Yt - Z*A;
    tmp = U/CSig';   % residuals standardized by chol(Sig)
    s2 = sum(tmp.^2, 2);
    h = sample_CSV_h_ARMH(s2, phi, sigh2, h, n, 30);

    % sample sigh2
    eh = [h(1)*sqrt(1-phi^2); h(2:end) - phi*h(1:end-1)];
    sigh2 = 1/gamrnd(nu_h + Tt/2, 1/(S_h + sum(eh.^2)/2));

    % sample phi via an independence-chain MH step
    Kphi = 1/Vphi + sum(h(1:end-1).^2)/sigh2;
    phihat = (phi0/Vphi + h(1:end-1)'*h(2:end)/sigh2)/Kphi;
    phic = phihat + randn/sqrt(Kphi);
    gphi = @(x) 0.5*log(1-x^2) - 0.5*(1-x^2)/sigh2*h(1)^2;
    if abs(phic) < 0.998 && exp(gphi(phic) - gphi(phi)) > rand
        phi = phic;
    end

    if isim > burnin
        isave = isim - burnin;
        % forecast h_{Tt+1} via the AR(1) and the implied Sig_{Tt+1}
        htp1 = phi*h(end) + sqrt(sigh2)*randn;
        mu_full = xtp1*A;
        mu_q = mu_full(targets);
        Sig_q = exp(htp1)*Sig(targets, targets);
        CSig_q = chol(Sig_q, 'lower');
        u = CSig_q\(yreal - mu_q)';
        lden_joint = -q/2*log(2*pi) - sum(log(diag(CSig_q))) - 0.5*(u'*u);
        dSig_q = diag(Sig_q)';
        lden = -0.5*log(2*pi*dSig_q) - 0.5*(yreal - mu_q).^2./dSig_q;
        store_mu(isave, :) = mu_q;
        store_lden(isave, :) = [lden lden_joint];
    end
end

% aggregate across draws
pf = mean(store_mu);
tmpmax = max(store_lden);
lpls = log(mean(exp(store_lden - tmpmax))) + tmpmax;
lpl = lpls(1:q);
lpl_joint = lpls(q+1);
end
