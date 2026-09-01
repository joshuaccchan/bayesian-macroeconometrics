function [store_beta, store_l, store_h, store_Sig_t, store_Q, store_S, ...
    store_sigh2, store_beta0, store_l0, store_h0] ...
        = estimate_TVPVAR(Y, p, prior, nsim, burnin)
% estimate_TVPVAR.m
% Implements a Gibbs sampler for the TVP-VAR with stochastic
% volatility of Primiceri (2005); see Section 13.2 of the textbook.
%
% Model
%   y_t = X_t * beta_t + eps_t,   eps_t ~ N(0, Sigma_t),
%   Sigma_t^{-1} = L_t' * D_t^{-1} * L_t,
%   D_t = diag(exp(h_{1t}), ..., exp(h_{nt})),
% where L_t is unit-lower-triangular and l_t collects its m=n(n-1)/2
% free elements stacked equation by equation
%     l_t = (l_{2,1,t}, l_{3,1,t}, l_{3,2,t}, ..., l_{n,n-1,t})'.
% State equations
%   beta_t = beta_{t-1} + u_t^beta,    u_t^beta ~ N(0, Q),
%   l_t    = l_{t-1}    + u_t^l,       u_t^l    ~ N(0, S),
%   h_{it} = h_{i,t-1}  + u_{it}^h,    u_{it}^h ~ N(0, sigh2_i),
% with full Q, block-diagonal S = diag(S_1,...,S_{n-1}), and
% Gaussian priors on (beta_0, l_0, h_0).
%
% Requires: SURform.m, SVRW.m
%
% Inputs
%   Y      : (T_all) x n data, with the FIRST p ROWS used only as
%            initial lags. Effective sample length is T = T_all - p.
%   p      : number of VAR lags.
%   prior  : struct with fields
%       beta0_mean (nk x 1), beta0_var  (nk x nk SPD)
%       l0_mean    (m  x 1), l0_var     (m  x m  SPD)
%       h0_mean    (n  x 1), h0_var     (n  x n  SPD)
%       Q_nu (scalar), Q_S0 (nk x nk SPD; full IW prior scale)
%       S_nu (1 x (n-1) vector), S_S0 (1 x (n-1) cell, S_S0{j} is j x j SPD)
%       sigh2_nu (n x 1), sigh2_S0 (n x 1)
%   nsim, burnin : MCMC settings.
%
% Outputs (post-burn-in)
%   store_beta  : nsim x (T*nk)
%   store_l     : nsim x (T*m)
%   store_h     : nsim x T x n
%   store_Sig_t : nsim x (n*n) x T   (empty if too large)
%   store_Q     : nsim x (nk*nk)
%   store_S     : nsim x (m*m)
%   store_sigh2 : nsim x n
%   store_beta0 : nsim x nk
%   store_l0    : nsim x m
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
% The random-walk precision on a stacked (T*r)-vector of states with inner
% covariance A is kron(HtH, inv(A)).
H   = speye(T) - sparse(2:T, 1:T-1, ones(T-1, 1), T, T);
HtH = H' * H;

% prior
beta0_mean = prior.beta0_mean(:);
iVbeta0    = prior.beta0_var \ speye(nk);
l0_mean    = prior.l0_mean(:);
iVl0       = prior.l0_var \ speye(m);
h0_mean    = prior.h0_mean(:);
iVh0       = prior.h0_var \ speye(n);
Q_nu       = prior.Q_nu;
Q_S0       = prior.Q_S0;
S_nu       = prior.S_nu;
S_S0       = prior.S_S0;
sigh2_nu   = prior.sigh2_nu(:);
sigh2_S0   = prior.sigh2_S0(:);

% indices of free entries below diagonal of L, eq-by-eq order:
L_id = nonzeros(tril(reshape(1:n^2, n, n), -1)');

% initialize chain at zero states (h = 0 implies sigma = 1) with tiny
% innovation covariances so the first beta draw is heavily smoothed.
l = zeros(T*m, 1);
h = zeros(T, n);
Q = 1e-4 * eye(nk);
sigh2 = 1e-4 * ones(n, 1);
S = cell(n-1, 1);
for ii = 1:n-1
    S{ii} = 1e-4 * eye(ii);
end

% initial conditions
beta0 = beta0_mean;
l0    = l0_mean;
h0    = h0_mean;

% storage
store_beta  = zeros(nsim, T*nk);
store_l     = zeros(nsim, T*m);
store_h     = zeros(nsim, T, n);
populate_Sig = (n*n*T*nsim < 5e7);
if populate_Sig
    store_Sig_t = zeros(nsim, n*n, T);
else
    store_Sig_t = [];
end
store_Q     = zeros(nsim, nk*nk);
store_S     = zeros(nsim, m*m);
store_sigh2 = zeros(nsim, n);
store_beta0 = zeros(nsim, nk);
store_l0    = zeros(nsim, m);
store_h0    = zeros(nsim, n);

% MCMC starts here
for isim = 1:nsim + burnin
    % sample beta
    iSig_big = build_iSig(l, h, n, T, L_id);
    iQ = Q \ eye(nk);
    HiQH = kron(HtH, iQ);
    rhs_b = zeros(T*nk, 1);
    rhs_b(1:nk) = iQ * beta0;
    XiSig = X' * iSig_big;
    Kbeta = HiQH + XiSig * X;
    Cb = chol(Kbeta, 'lower');
    beta_hat = Cb' \ (Cb \ (rhs_b + XiSig * y));
    beta = beta_hat + Cb' \ randn(T*nk, 1);

    % sample l
    eps_full = y - X * beta;
    U = reshape(eps_full, n, T)';
    E_big = build_E(U, n, T, L_id);
    iD = sparse(1:n*T, 1:n*T, exp(-reshape(h', n*T, 1)));
    % Block-diagonal S^{-1} (m x m), with blocks S_i^{-1}
    [iS_I, iS_J, iS_V] = blockdiag_inv_triplets(S);
    iSmat   = sparse(iS_I, iS_J, iS_V, m, m);
    HliSHl  = kron(HtH, iSmat);
    rhs_l = zeros(T*m, 1);
    rhs_l(1:m) = iSmat * l0;
    EiD = E_big' * iD;
    Kl = HliSHl + EiD * E_big;
    Cl = chol(Kl, 'lower');
    l_hat = Cl' \ (Cl \ (rhs_l + EiD * eps_full));
    l = l_hat + Cl' \ randn(T*m, 1);

    % sample h equation by equation (eps_full unchanged since beta fixed)
    Eorth = reshape(build_L(l, n, T, L_id) * eps_full, n, T)';
    ystar = log(Eorth.^2 + 1e-4);
    for ii = 1:n
        h(:, ii) = SVRW(ystar(:, ii), h(:, ii), h0(ii), sigh2(ii));
    end

    % sample beta0, l0, h0 (with diffuse priors); reuse iQ and iSmat
    Kb0 = iVbeta0 + iQ;
    Cb0 = chol(Kb0, 'lower');
    b0_hat = Cb0' \ (Cb0 \ (iVbeta0 * beta0_mean + iQ * beta(1:nk)));
    beta0 = b0_hat + Cb0' \ randn(nk, 1);

    Kl0 = iVl0 + iSmat;
    Cl0 = chol(Kl0, 'lower');
    l0_hat = Cl0' \ (Cl0 \ (iVl0 * l0_mean + iSmat * l(1:m)));
    l0 = l0_hat + Cl0' \ randn(m, 1);

    isigh2 = sparse(1:n, 1:n, 1./sigh2);
    Kh0 = iVh0 + isigh2;
    Ch0 = chol(Kh0, 'lower');
    h0_hat = Ch0' \ (Ch0 \ (iVh0 * h0_mean + isigh2 * h(1, :)'));
    h0 = h0_hat + Ch0' \ randn(n, 1);

    % sample Q, S, sigh2
    bmat = reshape(beta, nk, T);
    db = bmat - [beta0, bmat(:, 1:end-1)];
    Q = iwishrnd(Q_S0 + db*db', Q_nu + T);
    lmat = reshape(l, m, T);
    dl = lmat - [l0, lmat(:, 1:end-1)];
    off = 0;
    for ii = 1:n-1
        di = ii;
        idx = off+1 : off+di;
        DLi = dl(idx, :);
        S{ii} = iwishrnd(S_S0{ii} + DLi*DLi', S_nu(ii) + T);
        off = off + di;
    end

    dh = h - [h0'; h(1:end-1, :)];
    sigh2 = 1 ./ gamrnd(sigh2_nu + T/2, ...
        1 ./ (sigh2_S0 + 0.5 * sum(dh.^2, 1)'));
    
    if isim > burnin
        isave = isim - burnin;
        store_beta(isave, :) = beta';
        store_l(isave, :)    = l';
        store_h(isave, :, :) = h;
        store_Q(isave, :)    = Q(:)';
        Sfull = zeros(m, m);
        off = 0;
        for ii = 1:n-1
            di = ii;
            Sfull(off+1:off+di, off+1:off+di) = S{ii};
            off = off + di;
        end
        store_S(isave, :)     = Sfull(:)';
        store_sigh2(isave, :) = sigh2';
        store_beta0(isave, :) = beta0';
        store_l0(isave, :)    = l0';
        store_h0(isave, :)    = h0';
        if populate_Sig            
            L_big = build_L(l, n, T, L_id);
            D_sqrt = sparse(1:n*T, 1:n*T, exp(reshape(h', n*T, 1) / 2));
            M_big = L_big \ D_sqrt;    % M_t = L_t \ D_t^{1/2}
            Sig_big = M_big * M_big';  % blkdiag(Sigma_1, ..., Sigma_T)
            store_Sig_t(isave, :, :) = reshape(full(nonzeros(Sig_big)), n*n, T);
        end
    end

    if mod(isim, 1000) == 0
        fprintf('estimate_TVPVAR: iteration %d / %d\n', ...
                isim, nsim + burnin);
    end
end

end 

%======================================================================
function iSig_big = build_iSig(l, h, n, T, L_id)
% Assemble sparse blkdiag(L_t' D_t^{-1} L_t, t=1..T), size (n*T) x (n*T).
L_big = build_L(l, n, T, L_id);
iD = sparse(1:n*T, 1:n*T, exp(-reshape(h', n*T, 1)));
iSig_big = L_big' * iD * L_big;
end

%======================================================================
function L_big = build_L(l, n, T, L_id)
% Assemble sparse L_big = blkdiag(L_1, ..., L_T), size (n*T) x (n*T),
% where each L_t is unit lower-triangular with free entries given by
% l_t (the t-th block of length m of the stacked vector l).
m = n*(n-1)/2;

% Position of each free l-entry within an n x n block
r_pos = mod(L_id - 1, n) + 1;         % row in {2, ..., n}
c_pos = (L_id - r_pos) / n + 1;       % col in {1, ..., n-1}

% Off-diagonal triplets of L_big = blkdiag(L_1, ..., L_T)
t_off_n = (0:T-1)' * n;               % T x 1
I = t_off_n + r_pos';                 % T x m 
J = t_off_n + c_pos';                 % T x m
V = reshape(l, m, T)';                % T x m; row t is l_t'

% Sparse L_big: unit diagonal in each block plus the off-diagonal entries
diag_idx = (1:n*T)';
L_big = sparse([diag_idx; I(:)], [diag_idx; J(:)], ...
               [ones(n*T, 1); V(:)], n*T, n*T);
end

%======================================================================
function E_big = build_E(U, n, T, L_id)
% Assemble sparse E (n*T x m*T) block-diagonal with the n x m block
% E_t in slot t. E_t maps l_t to L_t * eps_t - eps_t, with row r_pos
% and column pos carrying the entry -eps_{c_pos(pos), t}.
m = n*(n-1)/2;

% Position of each free l-entry within an n x n block
r_pos = mod(L_id - 1, n) + 1;         % row in {2, ..., n}
c_pos = (L_id - r_pos) / n + 1;       % col in {1, ..., n-1}

% Triplets of E_big = blkdiag(E_1, ..., E_T)
t_off_n = (0:T-1)' * n;               % T x 1
t_off_m = (0:T-1)' * m;               % T x 1
I = t_off_n + r_pos';                 % T x m 
J = t_off_m + (1:m);                  % T x m
V = -U(:, c_pos);                     % T x m

E_big = sparse(I(:), J(:), V(:), n*T, m*T);
end

%======================================================================
function [I, J, V] = blockdiag_inv_triplets(S)
% Build triplets (I,J,V) for sparse block-diagonal inverse of cell S.
% S{i} is i x i SPD; total size m = sum_{i=1}^{n-1} i.
nblocks = numel(S);
total_nnz = 0;
for i = 1:nblocks
    total_nnz = total_nnz + i*i;
end
I = zeros(total_nnz, 1);
J = zeros(total_nnz, 1);
V = zeros(total_nnz, 1);
ptr = 0;
off = 0;
for i = 1:nblocks
    di = i;
    iSi = S{i} \ eye(di);
    [ii, jj] = ndgrid(off + (1:di), off + (1:di));
    nz = di*di;
    I(ptr+1 : ptr+nz) = ii(:);
    J(ptr+1 : ptr+nz) = jj(:);
    V(ptr+1 : ptr+nz) = iSi(:);
    ptr = ptr + nz;
    off = off + di;
end
end
