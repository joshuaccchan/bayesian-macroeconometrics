% linreg_t.m
% Gibbs sampler for an AR(2) model of US PCE inflation with Student-t
% errors. The model is
%   y_t = beta_1 + beta_2 y_{t-1} + beta_3 y_{t-2} + eps_t,
%   (eps_t | lam_t) ~ N(0, sig2*lam_t),  lam_t ~ IG(nu/2, nu/2),
% with independent normal-inverse-gamma priors on (beta, sig2) and a
% uniform prior on nu over (2, nu_ub). The block for nu uses a
% Griddy-Gibbs step via sample_nu_griddy.m.

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

% prior hyperparameters (indep normal and inverse-gamma)
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
iLam = sparse(1:T,1:T,1./lam);  

store_theta = zeros(nsim,k+2);  % [beta', sig2, nu]
store_lam = zeros(nsim,T);
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
    end
end
% Posterior summaries
theta_mean = mean(store_theta,1);
theta_lo  = quantile(store_theta,.025,1)';
theta_hi  = quantile(store_theta,.975,1)';

lam_mean = mean(store_lam,1)';
lam_lo   = quantile(store_lam,0.025,1)';
lam_hi   = quantile(store_lam,0.975,1)';

figure('Position',[100 100 800 300]);

% left panel: posterior of nu
subplot(1,2,1);
histogram(store_theta(:,end),30,...
    'FaceColor',0.8*[1 1 1], 'EdgeColor','k');
box off; set(gca,'FontSize',14);
xlabel('$\nu$','Interpreter','latex');
ylabel('Frequency','Interpreter','latex');

% right panel: lambda_t (log scale)
subplot(1,2,2); hold on;
    % time index: 1961Q1 to 2019Q4
tq = (datetime(1961,1,1) + calquarters(0:T-1))';
shaded_band(tq, lam_lo, lam_hi, 0.85);
plot(tq, lam_mean, 'k', 'LineWidth', 2);
set(gca,'YScale','log');   % log scale for lambda
box off;
set(gca,'FontSize',14);
xlim([datetime(1960,1,1) datetime(2020,1,1)]); 
ylabel('$\lambda_t$ (log scale)','Interpreter','latex');


