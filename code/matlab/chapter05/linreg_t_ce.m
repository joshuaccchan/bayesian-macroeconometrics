% linreg_t_ce.m
% Computes the log marginal likelihood of the AR(2) model with
% Student-t errors for US PCE inflation via the cross-entropy method.
% The importance density is a product of a multivariate normal density
% for beta and inverse-gamma densities for sig2 and nu, with parameters
% fitted to posterior draws from linreg_t.m. The importance sampling
% estimator is then computed using R independent draws from the fitted
% density. The seed is reset before the importance draws so the
% estimator is reproducible.

linreg_t;  % estimate the t model
rng(42);   % reset seed for importance sampling
R = 10000; % number of importance sampling draws
R = 20*ceil(R/20);   % ensure R divisible by 20
m = size(store_theta,2);
T = size(y,1);

% obtain parameters for the IS density
b_hat = mean(store_theta(:,1:m-2))';
B_hat = cov(store_theta(:,1:m-2)) + 1e-10*eye(m-2);
tmp = gamfit(1./store_theta(:,m-1));
gam1_hat = tmp(1); gam2_hat = 1./tmp(2);
tmp = gamfit(1./store_theta(:,m));
alp1_hat = tmp(1); alp2_hat = 1./tmp(2);

% obtain IS draws from the optimal density
theta_IS = zeros(R,m);
theta_IS(:,1:m-2) = repmat(b_hat',R,1) ...
    + (chol(B_hat,'lower')*randn(m-2,R))';
theta_IS(:,m-1) = 1./gamrnd(gam1_hat,1./gam2_hat,R,1);
theta_IS(:,m) = 1./gamrnd(alp1_hat,1./alp2_hat,R,1);

% construct the prior density
prior = @(b,s,n) lmvnpdf(b,beta0,iVbeta\speye(k)) ...
    + ligampdf(s,nu0,S0) + log(1/(nu_ub-2))...
    - 1e100*((n<2) || (n>nu_ub));

% construct the IS density
g_IS = @(b,s,n) lmvnpdf(b,b_hat,B_hat) ...
    + ligampdf(s,gam1_hat,gam2_hat) ...
    + ligampdf(n,alp1_hat,alp2_hat);
store_w = zeros(R,1);

for isim  = 1:R
    theta = theta_IS(isim,:)';
    beta = theta(1:m-2);
    sig2 = theta(m-1);
    nu = theta(m);
    e = y - X*beta;
    llike = T*(gammaln((nu+1)/2) - gammaln(nu/2) ...
        - 0.5*log(nu*pi*sig2)) ...
        - (nu+1)/2*sum(log(1 + (e.^2)/(sig2*nu)));
    store_w(isim) = llike + prior(beta,sig2,nu) ...
        - g_IS(beta,sig2,nu);
end
% point estimate from all R importance weights
maxw_all = max(store_w);
log_ml = log(mean(exp(store_w-maxw_all))) + maxw_all;

% batch estimates for the numerical standard error
W = reshape(store_w,R/20,20);
maxw = max(W);
bigml = log(mean(exp(W-repmat(maxw,R/20,1)),1)) + maxw;
ml_std = std(bigml)/sqrt(20);
fprintf('Log marginal likelihood: %.2f\n', log_ml);
fprintf('Numerical std. error:    %.2f\n', ml_std);
