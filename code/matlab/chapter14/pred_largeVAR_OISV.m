function [pf, lpl_joint, lpl, hyp, draws_mu] = pred_largeVAR_OISV(Y, Y0, Tt, p, ...
    targets, prior_type, kappa1, kappa2, kappa3, nsim, burnin)
% This function computes the one-step-ahead posterior predictive means
% and log predictive likelihoods of the target variables from the
% large VAR with order-invariant stochastic volatility (Section 14.3),
%     y_t = A' x_t + eps_t,    eps_t ~ N(0, Sig_t),
%     Sig_t^{-1} = B0' D_t^{-1} B0,
%     D_t = diag(exp(h_{1t}), ..., exp(h_{nt})),
% where B0 is an unrestricted (dense) n x n matrix and each
% log-volatility follows the zero-mean stationary AR(1) process
%     h_{it} = phi_i*h_{i,t-1} + u_{it}^h,   u_{it}^h ~ N(0, sigh2_i),
%     h_{i1} ~ N(0, sigh2_i/(1 - phi_i^2)).
%
% The priors on the VAR coefficients ('minn', 'hs', 'mahp') are as in
% pred_largeVAR_SV.m; each element of B0 has the prior N(b0_ij, 10)
% with prior mean the identity matrix.
%
% Inputs:
% Y:          T x n matrix of observations
% Y0:         p0 x n matrix of pre-sample observations (p0 >= p)
% Tt:         scalar; forecast origin (estimation uses Y(1:Tt,:), the
%             forecast target is Y(Tt+1,:))
% p:          scalar; lag order
% targets:    q x 1 vector of indices of the target variables
% prior_type: 'minn', 'minnH', 'hs', or 'mahp' (see pred_largeVAR_SV.m)
% kappa1:     scalar; intercept prior variance
% kappa2:     scalar; own-lag tightness ('minn') or initial value
% kappa3:     scalar; cross-lag tightness ('minn') or initial value
% nsim:       scalar; number of post-burnin MCMC draws
% burnin:     scalar; number of burnin draws
%
% Outputs:
% pf:        1 x q posterior predictive means of the targets
% lpl_joint: scalar; log of the joint predictive density of the q
%            targets at their realized values (log-mean-exp over draws)
% lpl:       1 x q marginal log predictive likelihoods of the targets
% hyp:       nsim x 3 saved hyperparameter draws [kappa2, kappa3,
%            theta2] (diagnostics): kappa2/kappa3 are the sampled
%            tightness under 'minnH'/'mahp', theta2 the horseshoe global
%            scale under 'hs'; unused entries stay at their input values
% draws_mu:  nsim x q predictive-mean draws, for comparing the mean and
%            median point forecast

[~, n] = size(Y);
k = 1 + n*p;
q = length(targets);
Yt = Y(1:Tt, :);
yreal = Y(Tt+1, targets);

% priors for the volatility parameters and B0
phi0 = 0.95;                  Vphi = 0.05^2;
nu_h = 3*ones(n, 1);          S_h = 0.05*(nu_h - 1);
b0_B0 = eye(n);               iV_B0 = eye(n)/10;

% construct the regressor matrix Z = [1, y_{t-1}', ..., y_{t-p}']
tmpY = [Y0(end-p+1:end, :); Yt];
Z = zeros(Tt, n*p);
for i = 1:p
    Z(:, (i-1)*n+1:i*n) = tmpY(p-i+1:end-i, :);
end
Z = [ones(Tt, 1) Z];

% Minnesota structure: beta0 = 0 and the constants C_ij (V_Minn with
% unit tightness); is_own/is_cross flag own- and cross-lag positions
[beta0, V_alpha] = Minn_indep(p, kappa1, kappa2, kappa3, Y0, Yt, 0);
[~, C_Minn] = Minn_indep(p, kappa1, 1, 1, Y0, Yt, 0);
is_int = false(n*k, 1);  is_own = false(n*k, 1);
for i = 1:n
    is_int((i-1)*k + 1) = true;
    for l = 1:p
        is_own((i-1)*k + 1 + (l-1)*n + i) = true;
    end
end
is_cross = ~is_int & ~is_own;
nlag = n*k - n;   % number of lag coefficients

% regressor row for forecasting Y(Tt+1,:)
xtp1 = [1, reshape(Yt(end:-1:end-p+1, :)', 1, n*p)];

% initialize the Markov chain
A = zeros(k, n);
ZZ = Z'*Z;
for i = 1:n
    iVai = 1./V_alpha((i-1)*k+1:i*k);
    Ki = ZZ; Ki(1:k+1:end) = Ki(1:k+1:end) + iVai';
    A(:, i) = Ki\(Z'*Yt(:, i));
end
U = Yt - Z*A;
Sig_hat = U'*U/Tt;
B0 = diag(1./sqrt(diag(Sig_hat)));
h = zeros(Tt, n);
phi = phi0*ones(n, 1);
sigh2 = 0.05*ones(n, 1);
    % global-local prior states
tau2 = ones(n*k, 1);  nu_tau = ones(n*k, 1);
theta2 = 0.01;  xi_theta = 1;
xi_k2 = 1;  xi_k3 = 1;

store_lden = zeros(nsim, q+1);
store_mu = zeros(nsim, q);
store_hyp = zeros(nsim, 3);   % [kappa2, kappa3, theta2] per saved draw
for isim = 1:nsim + burnin
    % update the prior covariance of beta under the global-local priors
    switch prior_type
        case 'hs'
            V_alpha(~is_int) = theta2*tau2(~is_int);
        case 'minnH'
            V_alpha(is_own) = kappa2*C_Minn(is_own);
            V_alpha(is_cross) = kappa3*C_Minn(is_cross);
        case 'mahp'
            V_alpha(is_own) = kappa2*tau2(is_own).*C_Minn(is_own);
            V_alpha(is_cross) = kappa3*tau2(is_cross).*C_Minn(is_cross);
    end

    % sample B0 row by row via the Waggoner-Zha-Villani scheme
    U = Yt - Z*A;
    B0 = sample_B0(B0, U, h, b0_B0, iV_B0);

    % sample beta equation by equation (Corollary 14.1)
    eh_inv = exp(-h);
    for i = 1:n
        A(:, i) = 0;
        Etil = (Yt - Z*A)*B0'; % row t is z_t' = (B0(y_t - A_{-i}'x_t))'
        bi = B0(:, i);
        w = eh_inv*(bi.^2);    % w_it = sum_j B0(j,i)^2 exp(-h_jt)
        c = (eh_inv.*Etil)*bi;
        iVai = 1./V_alpha((i-1)*k+1:i*k);
        Kai = Z'*(w.*Z);
        Kai(1:k+1:end) = Kai(1:k+1:end) + iVai';
        CKai = chol(Kai, 'lower');
        ai_hat = CKai'\(CKai\(iVai.*beta0((i-1)*k+1:i*k) + Z'*c));
        A(:, i) = ai_hat + CKai'\randn(k, 1);
    end

    % sample the log-volatilities equation by equation; each h_i is
    % recentered to mean zero, since its level is absorbed into the
    % scale of the corresponding row of B0
    Eorth = (Yt - Z*A)*B0';
    ystar = log(Eorth.^2 + 1e-4);
    for i = 1:n
        h(:, i) = SVAR1(ystar(:, i), h(:, i), 0, phi(i), sigh2(i));
        h(:, i) = h(:, i) - mean(h(:, i));
    end

    % sample the volatility parameters (phi, sigh2)
    [phi, sigh2] = sample_SVAR1para(h, phi, sigh2, phi0, Vphi, nu_h, S_h);

    % sample the global-local prior hyperparameters
    alp = A(:);
    switch prior_type
        case 'hs'
            b2 = alp(~is_int).^2;
            tau2(~is_int) = 1./gamrnd(1, 1./(1./nu_tau(~is_int) ...
                + b2/(2*theta2)));
            nu_tau(~is_int) = 1./gamrnd(1, 1./(1 + 1./tau2(~is_int)));
            theta2 = 1/gamrnd((nlag+1)/2, 1/(1/xi_theta ...
                + sum(b2./tau2(~is_int))/2));
            xi_theta = 1/gamrnd(1, 1/(1 + 1/theta2));
        case 'minnH'
            so = sum(alp(is_own).^2./C_Minn(is_own));
            sc = sum(alp(is_cross).^2./C_Minn(is_cross));
            kappa2 = 1/gamrnd((n*p+1)/2, 1/(1/xi_k2 + so/2));
            xi_k2 = 1/gamrnd(1, 1/(1 + 1/kappa2));
            kappa3 = 1/gamrnd((n*(n-1)*p+1)/2, 1/(1/xi_k3 + sc/2));
            xi_k3 = 1/gamrnd(1, 1/(1 + 1/kappa3));
        case 'mahp'
            kap = kappa2*is_own + kappa3*is_cross;
            b2 = alp(~is_int).^2;
            denom = kap(~is_int).*C_Minn(~is_int);
            tau2(~is_int) = 1./gamrnd(1, 1./(1./nu_tau(~is_int) ...
                + b2./(2*denom)));
            nu_tau(~is_int) = 1./gamrnd(1, 1./(1 + 1./tau2(~is_int)));
            so = sum(alp(is_own).^2./(tau2(is_own).*C_Minn(is_own)));
            sc = sum(alp(is_cross).^2./(tau2(is_cross).*C_Minn(is_cross)));
            kappa2 = 1/gamrnd((n*p+1)/2, 1/(1/xi_k2 + so/2));
            xi_k2 = 1/gamrnd(1, 1/(1 + 1/kappa2));
            kappa3 = 1/gamrnd((n*(n-1)*p+1)/2, 1/(1/xi_k3 + sc/2));
            xi_k3 = 1/gamrnd(1, 1/(1 + 1/kappa3));
    end

    if isim > burnin
        isave = isim - burnin;
        % forecast h_{Tt+1} via the AR(1) and the implied Sig_{Tt+1}
        htp1 = phi.*h(end, :)' + sqrt(sigh2).*randn(n, 1);
        mu_full = xtp1*A;
        mu_q = mu_full(targets);
        Sig_tp1 = (B0\diag(exp(htp1)))/B0';
        Sig_q = Sig_tp1(targets, targets);
        Sig_q = (Sig_q + Sig_q')/2;
        CSig_q = chol(Sig_q, 'lower');
        u = CSig_q\(yreal - mu_q)';
        lden_joint = -q/2*log(2*pi) - sum(log(diag(CSig_q))) - 0.5*(u'*u);
        dSig_q = diag(Sig_q)';
        lden = -0.5*log(2*pi*dSig_q) - 0.5*(yreal - mu_q).^2./dSig_q;
        store_mu(isave, :) = mu_q;
        store_lden(isave, :) = [lden lden_joint];
        store_hyp(isave, :) = [kappa2, kappa3, theta2];
    end
end

% aggregate across draws
pf = mean(store_mu);
tmpmax = max(store_lden);
lpls = log(mean(exp(store_lden - tmpmax))) + tmpmax;
lpl = lpls(1:q);
lpl_joint = lpls(q+1);
hyp = store_hyp;       % saved [kappa2 kappa3 theta2] draws (diagnostics)
draws_mu = store_mu;   % nsim x q predictive-mean draws (mean vs median)
end
