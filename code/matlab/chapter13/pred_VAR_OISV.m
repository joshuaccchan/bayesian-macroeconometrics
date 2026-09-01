function [pf, lpl] = pred_VAR_OISV(Yt, Z, xt, beta0, ...
    V_Minn, nsim, burnin, var_idx, yreal)
% pred_VAR_OISV.m
% Computes the one-step-ahead posterior predictive mean
% and log predictive likelihood of y_{t, var_idx} from the VAR with
% order-invariant stochastic volatility,
%     y_t = x_t' A  + eps_t,    eps_t ~ N(0, Sig_t),
%     Sig_t^{-1} = B_0' D_t^{-1} B_0,    D_t = diag(exp(h_{1t}),...,exp(h_{nt})),
%     h_{i,t} = phi_i*h_{i,t-1} + u_{i,t}^h,    u_{i,t}^h ~ N(0, sigh2_i),
%     h_{i,1} ~ N(0, sigh2_i/(1 - phi_i^2)),
% with the independent Minnesota prior beta ~ N(beta0, diag(V_Minn)),
% N(b_{0,i}, V_b) prior on each row b_i of B_0,
% TN_{(-1,1)}(phi_0, V_phi) prior on phi_i, and
% IG(nu_h, S_h) prior on each sigh2_i. Five-block Gibbs sampler
% following Section 13.1.3:
%   1) B_0 row by row via the Waggoner-Zha-Villani approach
%   2) beta given B_0 and h (Gaussian regression)
%   3) h_{i,1:T} via SVAR1 (stationary AR(1) auxiliary mixture sampler)
%   4) phi_i via independence-chain Metropolis-Hastings
%   5) sigh2_i ~ IG
%
% Requires: SURform2.m, sample_B0.m, SVAR1.m, sample_SVAR1para.m
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

% OI-SV-specific priors
b0_B0 = eye(n);   % prior mean: row i is e_i
iV_B0 = eye(n);   % prior precision of each row (V_b = I)
phi_0 = 0.9; V_phi = 0.2^2;
nu_h  = 3*ones(n, 1); S_h = 0.1*ones(n, 1);

% SUR-form regressors and time-stacked observations
X = SURform2(Z, n);
y = reshape(Yt', n*T, 1);

% initialize the Gibbs sampler at OLS
A = Z \ Yt;
beta = A(:);
E = Yt - Z*A;
h0 = log(diag(E'*E/T));
h  = repmat(h0', T, 1);
sigh2 = 0.1*ones(n, 1);
phi   = 0.9*ones(n, 1);
B0    = eye(n);

store_mu  = zeros(nsim, 1);
store_var = zeros(nsim, 1);

for isim = 1:nsim + burnin
    % sample B_0 row by row via Waggoner-Zha-Villani
    A = reshape(beta, k, n);
    E = Yt - Z*A;
    B0 = sample_B0(B0, E, h, b0_B0, iV_B0);

    % sample beta
    bigB0 = kron(speye(T), B0);
    iD  = sparse(1:T*n, 1:T*n, reshape(1./exp(h)',1,T*n));
    iSig  = bigB0'*iD*bigB0;
    XiSig = X'*iSig;
    Kbeta = iVbeta + XiSig*X;
    Cbeta = chol(Kbeta, 'lower');
    beta_hat = Cbeta' \ (Cbeta \ (iVbeta*beta0 + XiSig*y));
    beta = beta_hat + Cbeta' \ randn(n*k, 1);

    % sample h equation by equation 
    A = reshape(beta, k, n);
    E = Yt - Z*A;
    Eorth = E*B0';                           
    ystar = log(Eorth.^2 + 1e-4);
    for i = 1:n
        h(:, i) = SVAR1(ystar(:, i), h(:, i), 0, phi(i), sigh2(i));
    end

    % sample phi and sigh2
    [phi, sigh2] = sample_SVAR1para(h, phi, sigh2, ...
        phi_0, V_phi, nu_h, S_h);

    if isim > burnin
        isave = isim - burnin;
        mu_full = xt*A;
        % forecast h_{T+1} via stationary AR(1)
        h_tp1 = phi.*h(end, :)' + sqrt(sigh2).*randn(n, 1);
        iB0 = B0\eye(n);
        Sig_tp1 = iB0*diag(exp(h_tp1))*iB0';
        store_mu(isave)  = mu_full(var_idx);
        store_var(isave) = Sig_tp1(var_idx, var_idx);
    end
end

% aggregate across draws
pf = mean(store_mu);
log_w = -0.5*log(2*pi*store_var) ...
    - 0.5*(yreal - store_mu).^2./store_var;
lpl = log(mean(exp(log_w - max(log_w)))) + max(log_w);
end
