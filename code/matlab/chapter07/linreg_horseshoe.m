% linreg_horseshoe.m
% Gibbs sampler for the regression of US PCE inflation on an
% intercept and the first two lags of 16 macroeconomic and financial
% indicators, with a horseshoe prior on the 32 slope coefficients
% and a diffuse normal prior on the intercept. The half-Cauchy local
% and global scales tau_j and theta are represented via the latent
% inverse-gamma scale mixture of Makalic and Schmidt (2016), yielding
% inverse-gamma full conditionals for (tau_j^2, theta^2, lam_tau,
% lam_theta). A small floor is imposed on tau_j^2 * theta^2 to avoid
% numerical ill-conditioning in the Gaussian update for beta.
% Requires PCE_regression_data.csv.

clear; clc; rng(42);

nsim = 50000;
burnin = 1000;

% Load data
data =  readmatrix('PCE_regression_data.csv',...
    'Range', 'B2:Q241'); % 1960Q1-2019Q4
y_raw = data(:, 1);  
X_raw = data;
p = 2; % # of lags
[T0, m] = size(X_raw);
X = zeros(T0, m*p); % construct predictors
for j = 1:p
    X(:,(j-1)*m+1:j*m) = [nan(j,m); X_raw(1:end-j,:)];    
end
    % Drop initial rows with missing values from lagging
y = y_raw((p+1):T0);
X = X((p+1):T0, :);  
T = length(y);
X = [ones(T,1) X]; % add an intercept
k = m*p+1;

% prior hyperparameters
Vbeta0 = 100; % prior variance for the intercept
nu0 = 3; S0 = 1;
    
store_beta = zeros(nsim,k);
XtX = X'*X;
Xty = X'*y;

% initialize
beta_ols = (X'*X)\(X'*y);
sig2_ols = sum((y-X*beta_ols).^2)/T;
beta = beta_ols;
sig2 = sig2_ols;
lam_tau = 1./gamrnd(1/2,1,k-1,1);
lam_theta = 1/gamrnd(1/2,1);
tau2 = 1./gamrnd(1/2,lam_tau);
theta2 = 1/gamrnd(1/2,lam_theta);

for isim = 1:nsim+burnin
    % sample beta 
    var_slope = tau2 * theta2;    
    var_slope = max(var_slope, 1e-10);  % set lower bounds
    prior_prec = [1/Vbeta0; 1./var_slope];    
    Dbeta = (sparse(1:k,1:k,prior_prec) + XtX/sig2)\speye(k);    
    beta_hat = Dbeta*Xty/sig2;
    beta = beta_hat + chol(Dbeta,'lower')*randn(k,1);

    % sample sig2
    e = y - X*beta;
    S_hat = S0 + e'*e/2;
    sig2 = 1/gamrnd(nu0+T/2,1/S_hat); 
    
    % sample tau2
    tmp1 = 1./lam_tau + beta(2:end).^2/(2*theta2);
    tau2 = 1./gamrnd(1,1./tmp1);

    % sample theta2
    tmp2 = 1/lam_theta + sum(beta(2:end).^2./tau2)/2;
    theta2 = 1/gamrnd(k/2, 1./tmp2);

    % sample lam_tau, lam_theta
    lam_tau = 1./gamrnd(1,1./(1+1./tau2));
    lam_theta = 1/gamrnd(1,1/(1+1/theta2));

    % store the parameters
    if isim > burnin
        isave = isim-burnin;
        store_beta(isave,:) = beta';
    end
end
beta_mean = mean(store_beta(:,2:end))';      
beta_ci = quantile(store_beta(:,2:end),[.05 .95],1)';

gray = [0.5 0.5 0.5];
short_names = {'PCE inflation', 'Oil price', 'FFR', '10y yield', ...
    'Term spread', 'AAA-FFR spread', 'Real M2', 'Consumer credit', ...
    'UM sentiment', 'Cap. utilization', 'Real GDP', 'Real PCE', ...
    'Ind. production', 'Unemployment', 'Payrolls', 'Housing starts'};
ytick_labels = [strcat(short_names, ' (lag 1)'), ...
                strcat(short_names, ' (lag 2)')];

fig1 = figure('Position',[100 100 600 700]);
hold on; box off;
    % 0-line
plot([0 0], [0 k], '-', 'Color', gray, 'LineWidth', 1);
    % Credible intervals + means
for idx = 1:k-1
    plot([beta_ci(idx,1) beta_ci(idx,2)], [idx idx], '-', ...
         'Color', gray, 'LineWidth', 2);
    plot(beta_mean(idx), idx, 'o', ...
         'MarkerFaceColor', gray, 'MarkerEdgeColor', gray);
end
xlabel('$\beta_j$', 'Interpreter', 'latex');
ylim([0 k]);
set(gca, 'YTick', 1:k-1, 'YTickLabel', ytick_labels, ...
         'TickLabelInterpreter', 'tex', ...
         'FontSize', 11, 'LineWidth', 1);