function [pf, lpl_joint, lpl] = pred_largeVAR_FSV(Y, Y0, Tt, p, r, ...
    targets, prior_type, kappa1, kappa2, kappa3, nsim, burnin)
% This function computes the one-step-ahead posterior predictive means
% and log predictive likelihoods of the target variables from the
% large VAR with factor stochastic volatility (Section 14.4),
%     y_t = A' x_t + L*f_t + u_t,
%     u_t ~ N(0, D_t),   f_t ~ N(0, G_t),
%     D_t = diag(exp(h_{1t}), ..., exp(h_{nt})),
%     G_t = diag(exp(h_{n+1,t}), ..., exp(h_{n+r,t})),
% so that the error covariance matrix is Sig_t = L*G_t*L' + D_t. The
% loading matrix L is left unrestricted; the factors and loadings are
% then not separately identified, but Sig_t is, and it is all the
% predictive density requires. The r factor log-volatilities follow
% zero-mean stationary AR(1) processes (their scales are absorbed into
% the loadings), while the n idiosyncratic log-volatilities follow
% stationary AR(1) processes with free levels mu_i.
%
% The priors on the VAR coefficients ('minn', 'hs', 'mahp') are as 
% in pred_largeVAR_SV.m; each free loading has the prior N(0, 1).
%
% Inputs:
% Y:          T x n matrix of observations
% Y0:         p0 x n matrix of pre-sample observations (p0 >= p)
% Tt:         scalar; forecast origin (estimation uses Y(1:Tt,:), the
%             forecast target is Y(Tt+1,:))
% p:          scalar; lag order
% r:          scalar; number of factors
% targets:    q x 1 vector of indices of the target variables
% prior_type: 'minn', 'hs', or 'mahp'
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

[~, n] = size(Y);
k = 1 + n*p;
q = length(targets);
Yt = Y(1:Tt, :);
yreal = Y(Tt+1, targets);

% priors for the volatility parameters; the factor log-volatilities
% (last r elements) have their levels fixed at zero
mu0 = zeros(n+r, 1);            Vmu = 100*ones(n+r, 1);
phi0 = 0.98*ones(n+r, 1);       Vphi = 0.05^2*ones(n+r, 1);
nu_h = 3*ones(n+r, 1);          S_h = 0.1*(nu_h - 1);
free_mu = [true(n, 1); false(r, 1)];
V_l = 1;   % prior variance of each free loading

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
mu = zeros(n+r, 1);
ZZ = Z'*Z;
for i = 1:n
    iVai = 1./V_alpha((i-1)*k+1:i*k);
    Ki = ZZ; Ki(1:k+1:end) = Ki(1:k+1:end) + iVai';
    A(:, i) = Ki\(Z'*Yt(:, i));
    mu(i) = log(mean((Yt(:, i) - Z*A(:, i)).^2)/2);
end
h = [repmat(mu(1:n)', Tt, 1), zeros(Tt, r)];
phi = phi0;
sigh2 = 0.05*ones(n+r, 1);
L = [eye(r); 0.1*ones(n-r, r)];
F = zeros(Tt, r);
    % global-local prior states
tau2 = ones(n*k, 1);  nu_tau = ones(n*k, 1);
theta2 = 0.01;  xi_theta = 1;
xi_k2 = 1;  xi_k3 = 1;

store_lden = zeros(nsim, q+1);
store_mu = zeros(nsim, q);
for isim = 1:nsim + burnin
    % update the prior covariance of beta under the global-local priors
    switch prior_type
        case 'hs'
            V_alpha(~is_int) = theta2*tau2(~is_int);
        case 'mahp'
            V_alpha(is_own) = kappa2*tau2(is_own).*C_Minn(is_own);
            V_alpha(is_cross) = kappa3*tau2(is_cross).*C_Minn(is_cross);
    end

    % sample the factors f_1, ..., f_T jointly via the precision sampler
    e = reshape((Yt - Z*A)', Tt*n, 1);
    Xf = kron(speye(Tt), L);
    XfiSig = Xf'*sparse(1:Tt*n, 1:Tt*n, reshape(exp(-h(:, 1:n))', Tt*n, 1));
    Kf = sparse(1:Tt*r, 1:Tt*r, reshape(exp(-h(:, n+1:end))', Tt*r, 1)) ...
        + XfiSig*Xf;
    CKf = chol(Kf, 'lower');
    f_hat = CKf'\(CKf\(XfiSig*e));
    f = f_hat + CKf'\randn(Tt*r, 1);
    F = reshape(f, r, Tt)';

    % sample (beta_i, l_i) equation by equation: given the factors, the
    % n equations are independent Gaussian regressions
    Zi = [Z F];
    for i = 1:n
        wi = exp(-h(:, i));
        iVti = [1./V_alpha((i-1)*k+1:i*k); ones(r, 1)/V_l];
        ti0 = [beta0((i-1)*k+1:i*k); zeros(r, 1)];
        Kti = Zi'*(wi.*Zi);
        Kti(1:k+r+1:end) = Kti(1:k+r+1:end) + iVti';
        CKti = chol(Kti, 'lower');
        ti_hat = CKti'\(CKti\(iVti.*ti0 + Zi'*(wi.*Yt(:, i))));
        ti = ti_hat + CKti'\randn(k+r, 1);
        A(:, i) = ti(1:k);
        L(i, :) = ti(k+1:end)';
    end

    % sample the log-volatilities series by series
    U = Yt - Z*A - F*L';
    ystar = log([U F].^2 + 1e-4);
    for i = 1:n+r
        h(:, i) = SVAR1(ystar(:, i), h(:, i), mu(i), phi(i), sigh2(i));
    end

    % sample the volatility parameters (mu, phi, sigh2)
    [mu, phi, sigh2] = sample_SVAR1para_mu(h, mu, phi, sigh2, ...
        mu0, Vmu, phi0, Vphi, nu_h, S_h, free_mu);

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
        htp1 = mu + phi.*(h(end, :)' - mu) + sqrt(sigh2).*randn(n+r, 1);
        mu_full = xtp1*A;
        mu_q = mu_full(targets);
        Sig_tp1 = diag(exp(htp1(1:n))) + L*diag(exp(htp1(n+1:end)))*L';
        Sig_q = Sig_tp1(targets, targets);
        Sig_q = (Sig_q + Sig_q')/2;
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
