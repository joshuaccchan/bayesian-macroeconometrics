function [pf, lpl, k2m, k3m] = pred_VAR_SV_adapt(Yt, Z, xt, beta0, ...
    V_Minn, s2_hat, nsim, burnin, var_idx, yreal)
% pred_VAR_SV_adapt.m
% Adaptive-Minnesota version of pred_VAR_SV.m (Cholesky stochastic
% volatility). Identical five-block Gibbs sampler, plus an extra Gibbs
% step that estimates the Minnesota shrinkage hyperparameters kappa2
% (own-lag) and kappa3 (cross-lag) from their conjugate IG full
% conditionals; the intercept prior variance (kappa1) is held fixed.
%
% Requires: SURform2.m, SVRW.m
%
% Extra input:
% s2_hat:   n x 1 univariate AR(p) residual variances (from Minn_indep)
% Extra outputs:
% k2m, k3m: posterior means of kappa2, kappa3

[T, n] = size(Yt);
k = size(Z, 2);
p = (k - 1)/n;
m = n*(n - 1)/2;

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

% SV-specific priors
l0   = zeros(m, 1);          iVl  = speye(m);
m_h0 = zeros(n, 1);          iVh0 = speye(n)/10;
nu_h = 3*ones(n, 1);         S_h  = 0.1*ones(n, 1);

% indices of the strictly lower-triangular elements of L in row-major order
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
store_k2  = zeros(nsim, 1);
store_k3  = zeros(nsim, 1);

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

    % sample kappa2, kappa3 and rebuild V_Minn / iVbeta (intercepts fixed)
    b_own   = beta(own_idx)   - beta0(own_idx);
    b_cross = beta(cross_idx) - beta0(cross_idx);
    kappa2 = 1/gamrnd(nu_k2 + n_own/2,   1/(S_k2 + 0.5*sum(b_own.^2   ./ d_own)));
    kappa3 = 1/gamrnd(nu_k3 + n_cross/2, 1/(S_k3 + 0.5*sum(b_cross.^2 ./ d_cross)));
    V_Minn(own_idx)   = kappa2 .* d_own;
    V_Minn(cross_idx) = kappa3 .* d_cross;
    iVbeta = sparse(1:n*k, 1:n*k, 1./V_Minn);

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
