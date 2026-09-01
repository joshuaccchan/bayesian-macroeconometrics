% linreg_ssvs.m
% Collapsed Gibbs sampler for the point-mass spike-and-slab SSVS
% regression of US PCE inflation on an intercept and the first two
% lags of 16 macroeconomic and financial indicators. 
% Requires logpost_gam.m and PCE_regression_data.csv.

clear; clc; rng(42);

nsim = 20000;
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

    % priors
iVbeta = speye(k)/100;
nu0 = 3; S0 = 1;
bp = .5*ones(k,1);  % prior inclusion probabilities
   
store_gam = zeros(nsim,k);
store_betatilde = zeros(nsim,k);

    % initialize the Markov chain
beta_ols = (X'*X)\(X'*y);
sig2_ols = sum((y-X*beta_ols).^2)/T;
beta_ols_std = sqrt(sig2_ols*diag((X'*X)\speye(k)));
beta = beta_ols;
sig2 = sig2_ols;
gam = (abs(beta)./beta_ols_std > 1.65);    
gam = [1; gam(2:end)];   % fix gamma_1 = 1

for isim = 1:nsim+burnin
    % sample gamma marginal of beta (single-site)
    for j = 2:k
        % evaluate log-kernel at gamma_j = 0
        gam0 = gam;
        gam0(j) = 0;
        l0 = logpost_gam(gam0,y,X,sig2,bp,iVbeta);

        % evaluate log-kernel at gamma_j = 1
        gam1 = gam;
        gam1(j) = 1;
        l1 = logpost_gam(gam1,y,X,sig2,bp,iVbeta);

        % stable Bernoulli draw        
        mm = max(l0,l1);
        p1 = exp(l1 - mm)/(exp(l0 - mm) + exp(l1 - mm));
        gam(j) = (rand < p1);
    end
    Gam = sparse(1:k,1:k,gam);

    % sample beta | gamma, sig2        
    Xtilde = X*Gam;
    Dbeta = (iVbeta + Xtilde'*Xtilde/sig2)\speye(k);
    beta_hat = Dbeta*(Xtilde'*y/sig2);
    C = chol(Dbeta,'lower');
    beta = beta_hat + C*randn(k,1);
    
    % sample sig2
    e = y-Xtilde*beta;
    sig2 = 1/gamrnd(nu0+T/2,1/(S0+e'*e/2));
    
    if isim > burnin
        % store the parameters
        isave = isim-burnin;
        store_betatilde(isave,:) = (beta.*gam)';
        store_gam(isave,:) = gam';
    end
    
end
betatilde_mean = mean(store_betatilde(:,2:end))'; 
betatilde_ci = quantile(store_betatilde(:,2:end),...
    [.05 .95],1)';
gam_mean = mean(store_gam)';
    % model size - exclude the intercept indicator
sumgam_mean = mean(sum(store_gam(:,2:end),2)); 

gray = [0.5 0.5 0.5];
short_names = {'PCE inflation', 'Oil price', 'FFR', '10y yield', ...
    'Term spread', 'AAA-FFR spread', 'Real M2', 'Consumer credit', ...
    'UM sentiment', 'Cap. utilization', 'Real GDP', 'Real PCE', ...
    'Ind. production', 'Unemployment', 'Payrolls', 'Housing starts'};
ytick_labels = [strcat(short_names, ' (lag 1)'), ...
                strcat(short_names, ' (lag 2)')];

figure('Position',[100 100 600 700]);
hold on; box off;
    % 0-line
plot([0 0], [0 k], '-', 'Color', gray, 'LineWidth', 1);
    % Credible intervals + means
for idx = 1:k-1
    plot([betatilde_ci(idx,1) betatilde_ci(idx,2)], [idx idx], '-', ...
         'Color', gray, 'LineWidth', 2);
    plot(betatilde_mean(idx), idx, 'o', ...
         'MarkerFaceColor', gray, 'MarkerEdgeColor', gray);
end
xlabel('$\tilde{\beta}_j$', 'Interpreter', 'latex');
ylim([0 k]);
set(gca, 'YTick', 1:k-1, 'YTickLabel', ytick_labels, ...
         'TickLabelInterpreter', 'tex', ...
         'FontSize', 11, 'LineWidth', 1);