function [pf, lpl, k2m, k3m] = pred_VAR_homo_adapt(Yt, Z, xt, beta0, ...
    V_Minn, s2_hat, nsim, burnin, var_idx, yreal)
% pred_VAR_homo_adapt.m
% Adaptive-Minnesota version of pred_VAR_homo.m. Identical two-block
% Gibbs sampler for the homoskedastic VAR, except that the Minnesota
% shrinkage hyperparameters kappa2 (own-lag) and kappa3 (cross-lag) are
% ESTIMATED from the data via an additional Gibbs step. Given the base
% divisors d (d_own = 1/l^2, d_cross = s_i^2/(l^2 s_j^2)) so that
%   Var(own a_{l,ii})   = kappa2 * d_own,
%   Var(cross a_{l,ij})  = kappa3 * d_cross,
% and independent IG(nu_k, S_k) priors on kappa2, kappa3 with zero prior
% mean on the coefficients, the full conditionals are conjugate:
%   kappa2 | beta ~ IG(nu_k2 + n_own/2,   S_k2 + 0.5*sum_own  b_j^2/d_j),
%   kappa3 | beta ~ IG(nu_k3 + n_cross/2, S_k3 + 0.5*sum_cross b_j^2/d_j).
% The intercept prior variance (kappa1) is held fixed.
%
% Extra input:
% s2_hat:   n x 1 univariate AR(p) residual variances (from Minn_indep)
% Extra outputs:
% k2m, k3m: posterior means of kappa2, kappa3

[T, n] = size(Yt);
k = size(Z, 2);
p = (k - 1)/n;

% adaptive-Minnesota IG priors on kappa2, kappa3 (prior means 0.2^2, 0.2^2/4)
nu_k2 = 3; S_k2 = 2*0.2^2;
nu_k3 = 3; S_k3 = 2*0.2^2/4;

% own- vs cross-lag positions and base divisors (mirror Minn_indep index loop)
n_own   = n*p;
n_cross = n*(n-1)*p;
own_idx   = zeros(n_own, 1);   d_own   = zeros(n_own, 1);
cross_idx = zeros(n_cross, 1); d_cross = zeros(n_cross, 1);
co = 0; cc = 0; count = 1;
for i = 1:n
    count = count + 1;                 % intercept position (kappa1, fixed)
    for l = 1:p
        for j = 1:n
            if i == j
                co = co + 1; own_idx(co) = count;   d_own(co) = 1/l^2;
            else
                cc = cc + 1; cross_idx(cc) = count; d_cross(cc) = s2_hat(i)/(l^2*s2_hat(j));
            end
            count = count + 1;
        end
    end
end

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
store_k2  = zeros(nsim, 1);
store_k3  = zeros(nsim, 1);

for isim = 1:nsim + burnin
    % sample beta
    Kbeta = iVbeta + kron(iSig, ZZ);
    Cbeta = chol(Kbeta, 'lower');
    beta_hat = Cbeta' \ (Cbeta \ (iVbeta*beta0 + reshape(ZY*iSig, n*k, 1)));
    beta = beta_hat + Cbeta' \ randn(n*k, 1);

    % sample kappa2, kappa3 and rebuild V_Minn / iVbeta (intercepts fixed)
    b_own   = beta(own_idx)   - beta0(own_idx);
    b_cross = beta(cross_idx) - beta0(cross_idx);
    kappa2 = 1/gamrnd(nu_k2 + n_own/2,   1/(S_k2 + 0.5*sum(b_own.^2   ./ d_own)));
    kappa3 = 1/gamrnd(nu_k3 + n_cross/2, 1/(S_k3 + 0.5*sum(b_cross.^2 ./ d_cross)));
    V_Minn(own_idx)   = kappa2 .* d_own;
    V_Minn(cross_idx) = kappa3 .* d_cross;
    iVbeta = sparse(1:n*k, 1:n*k, 1./V_Minn);

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
        store_k2(isave)  = kappa2;
        store_k3(isave)  = kappa3;
    end
end

% aggregate across draws
pf = mean(store_mu);
log_w = -0.5*log(2*pi*store_var) - 0.5*(yreal - store_mu).^2./store_var;
lpl = log(mean(exp(log_w - max(log_w)))) + max(log_w);
k2m = mean(store_k2);
k3m = mean(store_k3);
end
