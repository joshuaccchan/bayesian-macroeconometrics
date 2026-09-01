function [store_beta, store_l, store_h, store_Sig_t, store_Q, ...
    store_sigh2, store_beta0, store_h0] ...
        = estimate_TVPVAR_constL(Y, p, prior, nsim, burnin)
% estimate_TVPVAR_constL.m
% Gibbs sampler for the TVP-VAR with stochastic volatility in which the
% lower-triangular factor L in Sigma_t^{-1} = L' * D_t^{-1} * L is held
% CONSTANT over time (L_t = L for all t). This is the Cogley-Sargent (2005)
% specification: only the VAR coefficients beta_t and the log-volatilities
% h_t are time-varying, while the contemporaneous relations L are constant.
%
% Model
%   y_t = X_t * beta_t + eps_t,   eps_t ~ N(0, Sigma_t),
%   Sigma_t^{-1} = L' * D_t^{-1} * L,     (L constant, unit lower triangular)
%   D_t = diag(exp(h_{1t}), ..., exp(h_{nt})),
% where l collects the m = n(n-1)/2 free elements of L stacked
% equation by equation  l = (l_{2,1}, l_{3,1}, l_{3,2}, ..., l_{n,n-1})'.
% State equations
%   beta_t = beta_{t-1} + u_t^beta,   u_t^beta ~ N(0, Q),
%   h_{it} = h_{i,t-1}  + u_{it}^h,   u_{it}^h ~ N(0, sigh2_i),
% with full Q and Gaussian priors on (beta_0, h_0). The constant l has a
% fixed Gaussian prior l ~ N(m_l, V_l).
%
% Relative to estimate_TVPVAR.m, Step 2 (time-varying l path with random-walk
% innovation covariance S) is replaced by a single constant-l Gaussian draw,
% and the S block / l_0 initial condition are removed. Blocks 1 (beta RW),
% 3 (h RW SV), 4 (beta_0, h_0), and the Q, sigh2 hyperparameter draws are
% unchanged from estimate_TVPVAR.m.
%
% Requires: SURform.m, SVRW.m
%
% Inputs
%   Y      : (T_all) x n data, with the FIRST p ROWS used only as
%            initial lags. Effective sample length is T = T_all - p.
%   p      : number of VAR lags.
%   prior  : struct with fields
%       beta0_mean (nk x 1), beta0_var  (nk x nk SPD)
%       l0_mean    (m  x 1), l0_var     (m  x m  SPD)   % used as (m_l, V_l)
%       h0_mean    (n  x 1), h0_var     (n  x n  SPD)
%       Q_nu (scalar), Q_S0 (nk x nk SPD; full IW prior scale)
%       sigh2_nu (n x 1), sigh2_S0 (n x 1)
%     (Fields S_nu, S_S0 are ignored if present.)
%   nsim, burnin : MCMC settings.
%
% Outputs (post-burn-in)
%   store_beta  : nsim x (T*nk)
%   store_l     : nsim x m           (constant l for each draw)
%   store_h     : nsim x T x n
%   store_Sig_t : []  (residual std devs recomputed from store_l, store_h)
%   store_Q     : nsim x (nk*nk)
%   store_sigh2 : nsim x n
%   store_beta0 : nsim x nk
%   store_h0    : nsim x n

[Tall, n] = size(Y);
k  = 1 + n*p;
nk = n*k;
m  = n*(n-1)/2;
T  = Tall - p;

% construct X and y
Yeff = Y(p+1:end, :);
X_tilde = zeros(T, n*p);
for i = 1:p
    X_tilde(:, (i-1)*n+1 : i*n) = Y(p-i+1 : end-i, :);
end
y = reshape(Yeff', n*T, 1);
X = SURform([ones(n*T, 1) kron(X_tilde, ones(n, 1))]);

% T x T first-difference operator H and its Gram matrix HtH = H'*H
H   = speye(T) - sparse(2:T, 1:T-1, ones(T-1, 1), T, T);
HtH = H' * H;

% prior
beta0_mean = prior.beta0_mean(:);
iVbeta0    = prior.beta0_var \ speye(nk);
l_mean     = prior.l0_mean(:);            % prior mean m_l of constant l
iVl        = prior.l0_var \ speye(m);     % prior precision V_l^{-1}
h0_mean    = prior.h0_mean(:);
iVh0       = prior.h0_var \ speye(n);
Q_nu       = prior.Q_nu;
Q_S0       = prior.Q_S0;
sigh2_nu   = prior.sigh2_nu(:);
sigh2_S0   = prior.sigh2_S0(:);

% indices of free entries below diagonal of L, eq-by-eq order:
L_id = nonzeros(tril(reshape(1:n^2, n, n), -1)');

% initialize chain at zero states (h = 0 implies sigma = 1) with tiny
% innovation covariances so the first beta draw is heavily smoothed.
l = l_mean;                 % constant l initialized at prior mean
h = zeros(T, n);
Q = 1e-4 * eye(nk);
sigh2 = 1e-4 * ones(n, 1);

% initial conditions
beta0 = beta0_mean;
h0    = h0_mean;

% storage
store_beta  = zeros(nsim, T*nk);
store_l     = zeros(nsim, m);
store_h     = zeros(nsim, T, n);
store_Sig_t = [];
store_Q     = zeros(nsim, nk*nk);
store_sigh2 = zeros(nsim, n);
store_beta0 = zeros(nsim, nk);
store_h0    = zeros(nsim, n);

% MCMC starts here
for isim = 1:nsim + burnin
    % replicate the constant l across all T blocks for the sparse builders
    l_rep = repmat(l, T, 1);

    % ---- Block 1: sample beta (unchanged) ----
    iSig_big = build_iSig(l_rep, h, n, T, L_id);
    iQ = Q \ eye(nk);
    HiQH = kron(HtH, iQ);
    rhs_b = zeros(T*nk, 1);
    rhs_b(1:nk) = iQ * beta0;
    XiSig = X' * iSig_big;
    Kbeta = HiQH + XiSig * X;
    Cb = chol(Kbeta, 'lower');
    beta_hat = Cb' \ (Cb \ (rhs_b + XiSig * y));
    beta = beta_hat + Cb' \ randn(T*nk, 1);

    % ---- Block 2: sample CONSTANT l ----
    % eps_t = E_t l + u_t, u_t ~ N(0, D_t), stacked over all t:
    %   (l | .) ~ N(lhat, Kl^{-1}),  Kl = V_l^{-1} + E' D^{-1} E,
    %   lhat = Kl^{-1} (V_l^{-1} m_l + E' D^{-1} eps).
    eps_full = y - X * beta;
    U = reshape(eps_full, n, T)';
    E_big = build_E_const(U, n, T, L_id);
    iD = sparse(1:n*T, 1:n*T, exp(-reshape(h', n*T, 1)));
    EiD = E_big' * iD;
    Kl  = iVl + EiD * E_big;
    Cl  = chol(Kl, 'lower');
    l_hat = Cl' \ (Cl \ (iVl * l_mean + EiD * eps_full));
    l = l_hat + Cl' \ randn(m, 1);

    % ---- Block 3: sample h equation by equation (unchanged) ----
    l_rep = repmat(l, T, 1);
    Eorth = reshape(build_L(l_rep, n, T, L_id) * eps_full, n, T)';
    ystar = log(Eorth.^2 + 1e-4);
    for ii = 1:n
        h(:, ii) = SVRW(ystar(:, ii), h(:, ii), h0(ii), sigh2(ii));
    end

    % ---- Block 4: sample beta0, h0 (l0 removed) ----
    Kb0 = iVbeta0 + iQ;
    Cb0 = chol(Kb0, 'lower');
    b0_hat = Cb0' \ (Cb0 \ (iVbeta0 * beta0_mean + iQ * beta(1:nk)));
    beta0 = b0_hat + Cb0' \ randn(nk, 1);

    isigh2 = sparse(1:n, 1:n, 1./sigh2);
    Kh0 = iVh0 + isigh2;
    Ch0 = chol(Kh0, 'lower');
    h0_hat = Ch0' \ (Ch0 \ (iVh0 * h0_mean + isigh2 * h(1, :)'));
    h0 = h0_hat + Ch0' \ randn(n, 1);

    % ---- Block 5: sample Q, sigh2 (S removed) ----
    bmat = reshape(beta, nk, T);
    db = bmat - [beta0, bmat(:, 1:end-1)];
    Q = iwishrnd(Q_S0 + db*db', Q_nu + T);

    dh = h - [h0'; h(1:end-1, :)];
    sigh2 = 1 ./ gamrnd(sigh2_nu + T/2, ...
        1 ./ (sigh2_S0 + 0.5 * sum(dh.^2, 1)'));

    if isim > burnin
        isave = isim - burnin;
        store_beta(isave, :) = beta';
        store_l(isave, :)    = l';
        store_h(isave, :, :) = h;
        store_Q(isave, :)    = Q(:)';
        store_sigh2(isave, :) = sigh2';
        store_beta0(isave, :) = beta0';
        store_h0(isave, :)    = h0';
    end

    if mod(isim, 1000) == 0
        fprintf('estimate_TVPVAR_constL: iteration %d / %d\n', ...
                isim, nsim + burnin);
    end
end

end

%======================================================================
function iSig_big = build_iSig(l, h, n, T, L_id)
% Assemble sparse blkdiag(L' D_t^{-1} L, t=1..T), size (n*T) x (n*T).
L_big = build_L(l, n, T, L_id);
iD = sparse(1:n*T, 1:n*T, exp(-reshape(h', n*T, 1)));
iSig_big = L_big' * iD * L_big;
end

%======================================================================
function L_big = build_L(l, n, T, L_id)
% Assemble sparse L_big = blkdiag(L_1, ..., L_T), size (n*T) x (n*T),
% where each L_t is unit lower-triangular with free entries given by
% the t-th block of length m of the stacked vector l.
m = n*(n-1)/2;

r_pos = mod(L_id - 1, n) + 1;         % row in {2, ..., n}
c_pos = (L_id - r_pos) / n + 1;       % col in {1, ..., n-1}

t_off_n = (0:T-1)' * n;               % T x 1
I = t_off_n + r_pos';                 % T x m
J = t_off_n + c_pos';                 % T x m
V = reshape(l, m, T)';                % T x m; row t is l_t'

diag_idx = (1:n*T)';
L_big = sparse([diag_idx; I(:)], [diag_idx; J(:)], ...
               [ones(n*T, 1); V(:)], n*T, n*T);
end

%======================================================================
function E_big = build_E_const(U, n, T, L_id)
% Assemble sparse E (n*T x m) mapping the CONSTANT l to the stacked
% n*T vector with block E_t in rows (t-1)*n+1 : t*n. All blocks share
% the SAME m columns (unlike the time-varying case), so E_t l = L*eps_t
% - eps_t. Row r_pos, column pos carries -eps_{c_pos(pos), t}.
m = n*(n-1)/2;

r_pos = mod(L_id - 1, n) + 1;         % row in {2, ..., n}
c_pos = (L_id - r_pos) / n + 1;       % col in {1, ..., n-1}

t_off_n = (0:T-1)' * n;               % T x 1
I = t_off_n + r_pos';                 % T x m
J = repmat(1:m, T, 1);                % T x m; columns fixed (no t offset)
V = -U(:, c_pos);                     % T x m

E_big = sparse(I(:), J(:), V(:), n*T, m);
end
