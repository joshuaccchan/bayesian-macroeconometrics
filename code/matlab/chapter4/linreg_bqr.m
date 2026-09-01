% linreg_bqr.m
% Gibbs sampler for Bayesian quantile regression of growth-at-risk.
% The model is
%   y_{t+h} = beta_1 + beta_2 gdp_t + beta_3 nfci_t + eps_{t+h},
%   eps_{t+h} ~ AL(0, sig2, tau),
% with the location-scale mixture-of-normals representation
% (eps | lam) ~ N(vartheta*lam, varphi*sig2*lam), lam ~ G(1, 1/sig2),
% and independent N + IG priors on (beta, sig2). The latent scales
% {lam_t} are updated via the inverse Gaussian representation; see
% igaussrnd.m.

clear; clc; rng(42);

nsim = 20000; burnin = 1000;
tau = 0.05;  % quantile level (e.g., 0.05, 0.10, 0.50)
h = 4;      % forecast horizon in quarters (e.g., 1 or 4)

% AL mixture constants
vartheta = (1 - 2*tau) / (tau*(1 - tau));
varphi = 2 / (tau*(1 - tau));

% load GDP-NFCI merged data
%   Date is a quarter-start date
%   GDP is quarterly (annualized) real GDP growth
%   NFCI is quarterly average of weekly NFCI
opts = detectImportOptions('GDP_NFCI_merged.csv');
opts = setvaropts(opts,'Date','InputFormat','MM/dd/uuuu');
TBL = readtable('GDP_NFCI_merged.csv', opts);

% keep observations with NFCI available
idx = ~isnan(TBL.NFCI) & ~isnan(TBL.GDP);
TBL = TBL(idx,:);
gdp  = TBL.GDP;
nfci = TBL.NFCI;

% Construct regression for Growth-at-Risk:
%   y_{t+h} = beta0 + beta1*gdp_t + beta2*nfci_t + eps_{t+h}
T0 = length(gdp);
T  = T0 - h;
y = gdp(1+h:end); % y_{t+h}
X = [ones(T,1), gdp(1:end-h), nfci(1:end-h)];
k = size(X,2);

% prior hyperparameters (indep normal and inverse-gamma)
beta0 = zeros(k,1);  
iVbeta = speye(k)/100;  % prior precision
nu0 = 3; 
S0 = 1*(nu0 - 1);

% initialize the Markov chain
beta = (X'*X)\(X'*y);
sig2 = mean((y - X*beta).^2);
lam = gamrnd(1, sig2, T, 1);
iLam   = sparse(1:T,1:T,1./lam);
store_theta = zeros(nsim, k+1);   % [beta' sig2]

% Gibbs sampler starts here
for isim = 1:nsim + burnin

    % sample beta
    ytilde = y - vartheta*lam;
    Dbeta = (iVbeta + (X'*iLam*X)/(varphi*sig2))\speye(k);
    beta_hat = Dbeta*(iVbeta*beta0 ...
        + (X'*iLam*ytilde)/(varphi*sig2));
    C = chol(Dbeta,'lower');
    beta = beta_hat + C*randn(k,1);

    % sample sig2
    e = y - X*beta - vartheta*lam;    
    S_hat = S0 + sum(lam) + e'*iLam*e/(2*varphi);
    sig2  = 1/gamrnd(nu0 + 3*T/2, 1/S_hat);

    % sample lambda    
    a_tau = (vartheta^2 + 2*varphi)/(varphi*sig2);
    mu = sqrt(vartheta^2 + 2*varphi) ./ abs(y - X*beta);
    lam = 1./igaussrnd(a_tau*ones(T,1), mu);
    iLam = sparse(1:T,1:T,1./lam);

    if isim > burnin
        isave = isim - burnin;
        store_theta(isave,:) = [beta' sig2];
    end
end
theta_mean = mean(store_theta,1)';
theta_CI  = quantile(store_theta,[.025 .975]);

disp('Posterior mean (beta; sig2):');
disp(theta_mean);
disp('Posterior 95% CI (rows: 2.5%, 97.5%):');
disp(theta_CI);

% compute GaR_t(h; tau) 
B = store_theta(:,1:k);
Xgar = X; % regressors at time t (aligned with y_{t+h})
GaR_draws = Xgar * B';  %  dimension is T x nsim
GaR_mean = mean(GaR_draws, 2); 

% Credible bands (pointwise)
GaR_lo95 = quantile(GaR_draws, 0.025, 2);
GaR_hi95 = quantile(GaR_draws, 0.975, 2);

% align dates with forecasted outcome y_{t+h}
date_f = datetime(TBL.Date(1+h:end), 'InputFormat', 'MM/dd/yyyy');

figure; 
hold on;
plot(date_f, GaR_mean, 'k-', 'LineWidth', 2);
hold off; box off;
ylabel('GaR');
set(gca, 'FontSize', 14);

% Start two quarters before first forecast date
x_start = date_f(1) - calquarters(2);
x_end   = date_f(end);
xlim([x_start x_end]);
