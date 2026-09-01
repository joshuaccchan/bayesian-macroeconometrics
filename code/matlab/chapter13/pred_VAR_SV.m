function [pf, lpl] = pred_VAR_SV(Yt, Z, xt, beta0, V_Minn, nsim, burnin, var_idx, yreal)
% pred_VAR_SV.m
% Computes the one-step-ahead posterior predictive mean
% and log predictive likelihood of y_{t, var_idx} from the VAR with
% Cholesky stochastic volatility,
%     y_t = x_t' A  + eps_t,    eps_t ~ N(0, Sig_t),
%     Sig_t^{-1} = L' D_t^{-1} L,    D_t = diag(exp(h_{1t}),...,exp(h_{nt})),
%     h_{i,t} = h_{i,t-1} + u_{i,t}^h,    u_{i,t}^h ~ N(0, sigh2_i),
% with the independent Minnesota prior beta ~ N(beta0, diag(V_Minn)),
% N(l0, V_l) prior on the free elements l of L,
% IG(nu_h, S_h) prior on each sigh2_i, and
% N(m_h0, V_h0) prior on h_0 = (h_{1,0},...,h_{n,0})'. Five-block Gibbs
% sampler following Section 13.1.2.
%
% Requires: SURform2.m, SVRW.m
%
% Inputs:
% Yt:       T x n matrix of observations up to time t-1
% Z:        T x k matrix of regressors (intercept + p lags), k = 1+n*p
% xt:       1 x k regressor row for forecasting y_t
% beta0:    (nk) x 1 Minnesota prior mean of beta
% V_Minn:   (nk) x 1 diagonal entries of the Minnesota prior covariance
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
m = n*(n - 1)/2;
iVbeta = sparse(1:n*k, 1:n*k, 1./V_Minn);

% SV-specific priors
l0   = zeros(m, 1);          iVl  = speye(m);
m_h0 = zeros(n, 1);          iVh0 = speye(n)/10;
nu_h = 3*ones(n, 1);         S_h  = 0.1*ones(n, 1);

% indices of the strictly lower-triangular elements of L in row-major
% order, so that L(L_id) = l_vec corresponds to
%   l_vec = (l_{21}, l_{31}, l_{32}, l_{41}, l_{42}, l_{43}, ...)'
L_id = zeros(m, 1);
ii = 0;
for i = 2:n
    for j = 1:i-1
        ii = ii + 1;
        L_id(ii) = i + (j-1)*n;
    end
end

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
l_vec = zeros(m, 1);
L = eye(n);

store_mu  = zeros(nsim, 1);
store_var = zeros(nsim, 1);

for isim = 1:nsim + burnin
    % sample beta
    L(L_id) = l_vec;
    bigL = kron(speye(T), L);
    iD = sparse(1:T*n, 1:T*n, reshape(1./exp(h)', 1, T*n));
    iSig = bigL'*iD*bigL;
    XiSig = X'*iSig;
    Kbeta = iVbeta + XiSig*X;
    Cbeta = chol(Kbeta, 'lower');
    beta_hat = Cbeta' \ (Cbeta \ (iVbeta*beta0 + XiSig*y));
    beta = beta_hat + Cbeta' \ randn(n*k, 1);

    % sample h equation by equation
    A = reshape(beta, k, n);
    E = Yt - Z*A; 
    Eorth = E*L';
    ystar = log(Eorth.^2 + 1e-4);
    for i = 1:n
        h(:, i) = SVRW(ystar(:, i), h(:, i), h0(i), sigh2(i));
    end

    % sample l_vec from the regression eps_t = E_t * l + eta_t
    Em = zeros(T*n, m);
    cE = 0;
    for ii = 1:n-1
        Em(ii+1:n:end, cE+1:cE+ii) = -E(:, 1:ii);
        cE = cE + ii;
    end
    iD = sparse(1:T*n, 1:T*n, reshape(1./exp(h)', 1, T*n));
    Kl = iVl + Em'*iD*Em;
    Cl = chol(Kl, 'lower');
    l_hat = Cl' \ (Cl \ (iVl*l0 + Em'*iD*reshape(E', T*n, 1)));
    l_vec = l_hat + Cl' \ randn(m, 1);

    % sample sigh2
    e2 = (h - [h0'; h(1:end-1, :)]).^2;
    sigh2 = 1./gamrnd(nu_h + T/2, 1./(S_h + sum(e2)'/2));

    % sample h0
    Kh0 = iVh0 + sparse(1:n, 1:n, 1./sigh2);
    Ch0 = chol(Kh0, 'lower');
    h0_hat = Ch0' \ (Ch0 \ (iVh0*m_h0 + h(1, :)'./sigh2));
    h0 = h0_hat + Ch0' \ randn(n, 1);

    if isim > burnin
        isave = isim - burnin;
        mu_full = xt*A;
        % forecast h_{T+1} via random walk and the implied Sig_{T+1}
        h_tp1 = h(end, :)' + sqrt(sigh2).*randn(n, 1);
        L(L_id) = l_vec;
        invL = L\eye(n);
        Sig_tp1 = invL*diag(exp(h_tp1))*invL';
        store_mu(isave)  = mu_full(var_idx);
        store_var(isave) = Sig_tp1(var_idx, var_idx);
    end
end

% aggregate across draws
pf = mean(store_mu);
log_w = -0.5*log(2*pi*store_var) - 0.5*(yreal - store_mu).^2./store_var;
lpl = log(mean(exp(log_w - max(log_w)))) + max(log_w);
end
