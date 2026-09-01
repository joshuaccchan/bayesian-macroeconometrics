% linreg_ma1_dic.m
% Computes the DIC for the AR(2) model with MA(1) errors (ARMA(2,1))
% for US PCE inflation. The deviance is stored at each post-burn-in
% Metropolis-within-Gibbs iteration; the DIC equals the posterior
% mean deviance plus the effective number of parameters.

clear; clc; rng(42);
nsim = 50000; burnin = 1000;

% load data
data = readmatrix('USPCE.csv', 'Range', 'B2:B241');
y0 = data(1:4);  % initial conditions
y = data(5:end); % sample used for estimation
T = size(y,1);

% regressors for AR(2): [1, y_{t-1}, y_{t-2}]
xlag1 = [y0(4); y(1:end-1)];  
xlag2 = [y0(3:4); y(1:end-2)];
X = [ones(T,1), xlag1, xlag2];
k = size(X,2);   % number of regressors

% prior hyperparameters (indep normal and inverse-gamma)
beta0 = zeros(k,1);  iVbeta = 1/100*speye(k); 
nu0 = 4;  S0 = 1;

% initialize the Markov chain
psi = 0;
beta = (X'*X)\(X'*y);
sig2 = sum((y-X*beta).^2)/T;

Hpsi = spdiags([psi*ones(T,1),ones(T,1)],[-1, 0],T,T);
store_theta = zeros(nsim,k+2);  % [beta', sig2, psi]
store_dev = zeros(nsim,1);
count_psi = 0;
g_var = 0.02;  % proposal variance for RW-MH step for psi

for isim = 1:nsim + burnin
    % sample beta    
    X_tilde = Hpsi\X; 
    y_tilde = Hpsi\y;    
    Dbeta = (iVbeta + X_tilde'*X_tilde/sig2)\speye(k); 
    beta_hat = Dbeta*(iVbeta*beta0 ...
        + X_tilde'*y_tilde/sig2);
    beta = beta_hat + chol(Dbeta,'lower')*randn(k,1);

    % sample sig2
    e = y - X*beta;
    u = Hpsi\e;
    sig2 = 1/gamrnd(nu0+T/2,1/(S0 + u'*u/2)); 
    
    % sample psi    
    [psi, accept] = sample_psi_RW(psi,e,sig2,g_var);
    Hpsi = spdiags([psi*ones(T,1),ones(T,1)],[-1, 0],T,T);    
    
    % store the parameters
    if isim > burnin
        isave = isim - burnin;
        count_psi = count_psi + accept; % post-burnin
        store_theta(isave,:) = [beta', sig2, psi];    
        store_dev(isave) = -2*loglike_MA1(psi,y-X*beta,sig2);      
    end
end

% acceptance rate
accept_rate  = count_psi/nsim;
fprintf('Acceptance rate for psi: %.2f\n', accept_rate);

theta_hat = mean(store_theta)';
beta_hat = theta_hat(1:3);
sig2_hat = theta_hat(4);
psi_hat = theta_hat(5);
pD = mean(store_dev) + 2*loglike_MA1(psi_hat,y-X*beta_hat,sig2_hat);
DIC = mean(store_dev) + pD;
fprintf('DIC: %.2f\n', DIC);
fprintf('pD: %.1f\n', pD);