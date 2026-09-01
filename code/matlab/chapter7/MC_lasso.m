% MC_lasso.m
% Monte Carlo benchmark for the Bayesian Lasso of Example 7.2:
% generates R = 100 datasets of size T = 100 with k regressors and
% true coefficient vector (1, 1, 0, ..., 0)', then compares the
% posterior-mean MSE of the Bayesian Lasso against ordinary least
% squares and ridge regression. The shrinkage parameter lambda is
% set as lambda = k * sqrt(sig2_LS) / sum |beta_LS|. Requires
% fit_BayesLasso.m, fit_BayesRidge.m, and igaurnd.m.

clear; clc; rng(42);

nsim = 5000;
burnin = 1000;
R = 100;
T = 100;
k = 20;  % number of regressors (change to 50 for the high-dim case)

% priors
nu0 = 5; S0 = 1;

% true parameter values
truebeta = [1 1 zeros(1,k-2)]';
truesig2 = 1;

store_MSE = zeros(R,3);
for idata = 1:R
    % generate data
    X = 5*rand(T,k);
    y = X*truebeta + sqrt(truesig2)*randn(T,1);

    beta_ols = (X'*X)\(X'*y);
    sig2_ols = sum((y-X*beta_ols).^2)/T;
    lambda = k*sqrt(sig2_ols)/sum(abs(beta_ols));

    beta_ridge = fit_BayesRidge(y,X,2*k*sig2_ols/(beta_ols'*beta_ols));
    beta_lasso = fit_BayesLasso(y,X,lambda,S0,nu0,nsim,burnin);

    store_MSE(idata,:) = [mean((truebeta-beta_ols).^2), ...
                          mean((truebeta-beta_ridge).^2), ...
                          mean((truebeta-beta_lasso).^2)];
end

fprintf('\nMedian MSE  (k = %d):\n', k);
fprintf('  OLS    %.4f\n  Ridge  %.4f\n  Lasso  %.4f\n', median(store_MSE));

fig1 = figure('Position',[100 100 400 600]);
boxplot(store_MSE,'labels',{'LS','Ridge','Lasso'},'Whisker',10, ...
    'Colors','k','Symbol','k+');
box off;
