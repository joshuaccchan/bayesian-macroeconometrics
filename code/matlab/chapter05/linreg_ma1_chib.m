% linreg_ma1_chib.m
% Computes the log marginal likelihood of the ARMA(2,1) model for US
% PCE inflation via Chib's method. The posterior ordinate at the
% posterior mean (beta*, sig2*, psi*) is factored as
%   p(beta*, sig2*, psi* | y) = p(beta* | y) * p(sig2* | y, beta*)
%                               * p(psi* | y, beta*, sig2*).
% The first factor is estimated from the main Gibbs run, the second
% from a reduced run that fixes beta at beta* and updates (sig2, psi),
% and the third by evaluating the conditional posterior of psi on a
% fine grid. The seed is reset before the reduced run so the marginal
% likelihood estimate is reproducible.

linreg_ma1;  % estimate the ARMA(2,1) model

nsim_re = 5000; % size of reduced runs
[nsim,m] = size(store_theta);
T = size(y,1);

theta_hat = mean(store_theta)';
beta_s = theta_hat(1:m-2); % beta*
sig2_s = theta_hat(m-1);   % sig2*
psi_s = theta_hat(m);      % psi*

% log likelihood at theta*
llike = loglike_MA1(psi_s, y - X*beta_s, sig2_s);

% log prior at theta*
log_prior=@(b,s,p) lmvnpdf(b,beta0,iVbeta\speye(k))...
    + ligampdf(s,nu0,S0) + log(1/2);

% 1) posterior of beta at beta_s using main run
store_lpbeta = zeros(nsim,1);
for isim = 1:nsim
    sig2 = store_theta(isim,m-1); % from the main run
    psi = store_theta(isim,m);
    Hpsi = speye(T) ...
        + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T);
    X_tilde = Hpsi\X; y_tilde = Hpsi\y;
    Dbeta = (iVbeta + X_tilde'*X_tilde/sig2)\speye(k);
    beta_hat = Dbeta * (iVbeta*beta0 ...
        + X_tilde'*y_tilde/sig2);
    store_lpbeta(isim) = ...
        lmvnpdf(beta_s,beta_hat,Dbeta);
end
a = max(store_lpbeta);
lpbeta = log(mean(exp(store_lpbeta-a))) + a;

% 2) posterior of sig2 at sig2_s using a reduced run
rng(42); % re-seed for reproducible reduced run
beta = beta_s; % fix beta at beta_s
e = y - X*beta;
sig2 = sig2_s;
psi = psi_s;
store_lpsig2 = zeros(nsim_re,1);
for isim = 1:nsim_re
    % sample psi
    psi = sample_psi_RW(psi,e,sig2,g_var);
    Hpsi = speye(T) ...
        + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T);
    % sample sig2
    tmp = Hpsi\e;
    sig2 = 1/gamrnd(nu0+T/2,1/(S0 + tmp'*tmp/2));

    store_lpsig2(isim,:) = ...
        ligampdf(sig2_s,nu0+T/2,S0 + tmp'*tmp/2);
end
a = max(store_lpsig2);
lpsig2 = log(mean(exp(store_lpsig2-a))) + a;

% 3) posterior of psi at psi_s via grid
n_grid = 399;
psi_grid = linspace(-.99,.99,n_grid)';
    % ensure psi_s is on the grid
psi_grid = sort([psi_grid;psi_s]);
n_grid = size(psi_grid,1);
idx_psi = (psi_grid==psi_s); % index for psi_s
lp_psi = zeros(n_grid,1);  % log posterior density
for igrid = 1:n_grid
    psi = psi_grid(igrid);
    lp_psi(igrid) = loglike_MA1(psi,y-X*beta_s,sig2_s);
end
p_psi = exp(lp_psi-max(lp_psi));
p_psi = p_psi/trapz(psi_grid,p_psi);
lppsi = log(p_psi(idx_psi));

log_ml = llike + log_prior(beta_s,sig2_s,psi_s)...
    - (lpbeta + lpsig2 + lppsi);

fprintf('Log marginal likelihood: %.2f\n', log_ml);
