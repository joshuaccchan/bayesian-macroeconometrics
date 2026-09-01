% linreg_ma1.m
% Metropolis-within-Gibbs sampler for an ARMA(2,1) model of US PCE
% inflation. The model is
%   y_t = beta_1 + beta_2 y_{t-1} + beta_3 y_{t-2} + eps_t,
%   eps_t = u_t + psi*u_{t-1},  u_t ~ N(0, sig2),
% with independent normal-inverse-gamma priors on (beta, sig2) and a
% uniform prior on psi over (-1, 1). The block for psi uses a
% random-walk Metropolis-Hastings step via sample_psi_RW.m.

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
    end
end

% acceptance rate
accept_rate  = count_psi/nsim;
fprintf('Acceptance rate for psi: %.2f\n', accept_rate);

% Posterior summaries
theta_mean = mean(store_theta,1);
theta_lo  = quantile(store_theta,.025,1)';
theta_hi  = quantile(store_theta,.975,1)';

figure('Position',[100 100 800 300]);

% left panel: posterior of psi
subplot(1,2,1);
histogram(store_theta(:,end),30,...
    'FaceColor',0.8*[1 1 1], 'EdgeColor','k');
box off; set(gca,'FontSize',14);
xlabel('$\psi$','Interpreter','latex');
ylabel('Frequency','Interpreter','latex');

% right panel: trace plot of psi
subplot(1,2,2);
plot(store_theta(:,end), 'k', 'LineWidth', 1);
box off; set(gca,'FontSize',14);
xlabel('Iteration','Interpreter','latex');
ylabel('$\psi$','Interpreter','latex');
xlim([0,nsim]);
