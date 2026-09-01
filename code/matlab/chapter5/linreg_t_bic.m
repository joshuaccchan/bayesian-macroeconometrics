% linreg_t_bic.m
% Computes the BIC for the AR(2) model with Student-t errors for US
% PCE inflation. The log-likelihood is stored at each post-burn-in
% Gibbs iteration, and the BIC is computed using the maximum sampled
% log-likelihood as a stand-in for the maximum likelihood value.

clear; clc; rng(42);
nsim = 20000; burnin = 1000;

% load data
data = readmatrix('USPCE.csv', 'Range', 'B2:B241');
y0 = data(1:4);  % initial conditions: [y_{-3},y_{-2},y_{-1},y_0]
y = data(5:end); % sample used for estimation
T = size(y,1);

% regressors for AR(2): [1, y_{t-1}, y_{t-2}]
xlag1 = [y0(4); y(1:end-1)];  
xlag2 = [y0(3:4); y(1:end-2)];
X = [ones(T,1), xlag1, xlag2];
k = size(X,2);   % number of regressors

% prior hyperparameters (independent normal and inverse-gamma)
beta0  = zeros(k,1);
iVbeta  = 1/100*speye(k);  % prior precision of beta
nu0    = 4;
S0     = 1;
nu_ub = 50;                % prior upperbound of nu 

% initialize the Markov chain
nu = 5;
beta = (X'*X)\(X'*y);
sig2 = sum((y-X*beta).^2)/T;
lam = ones(T,1);
iLam = sparse(1:T,1:T,1./lam);  % sparse version of Lambda^{-1}

store_theta = zeros(nsim,k+2);  % [beta', sig2, nu]
store_lam = zeros(nsim,T);
store_llike = zeros(nsim,1);
n_grid = 500;

for isim = 1:nsim + burnin
    % sample beta    
    Dbeta = (iVbeta + X'*iLam*X/sig2)\speye(k); 
    beta_hat = Dbeta*(iVbeta*beta0 + X'*iLam*y/sig2);
    C = chol(Dbeta,'lower');
    beta = beta_hat + C*randn(k,1);

    % sample sig2    
    e = y - X*beta;
    sig2 = 1/gamrnd(nu0+T/2,1/(S0 + e'*iLam*e/2)); 
        
    % sample lam    
    lam = 1./gamrnd((nu+1)/2,2./(nu+e.^2/sig2));
    iLam = sparse(1:T,1:T,1./lam);    
    
    % sample nu
    nu = sample_nu_griddy(lam,nu_ub,n_grid);
    
        % store the parameters
    if isim > burnin
        isave = isim - burnin;
        store_theta(isave,:) = [beta' sig2 nu];  
        store_lam(isave,:) = lam';
        llike = T*(gammaln((nu+1)/2) -gammaln(nu/2) -.5*log(nu*pi*sig2))...
            -(nu+1)/2*sum(log(1 + (y-X*beta).^2/(sig2*nu)));
        store_llike(isave,:) = llike;
    end
end
% Posterior summaries
theta_mean = mean(store_theta,1);
theta_lo  = quantile(store_theta,.025,1)';
theta_hi  = quantile(store_theta,.975,1)';

lam_mean = mean(store_lam,1)';
lam_lo   = quantile(store_lam,0.025,1)';
lam_hi   = quantile(store_lam,0.975,1)';

max_llike = max(store_llike);
BIC = -2*max_llike + 5*log(T);
fprintf('BIC: %.2f\n', BIC);
