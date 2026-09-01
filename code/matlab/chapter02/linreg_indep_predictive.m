% linreg_indep_predictive.m
% Gibbs sampler for an AR(2) model of US PCE inflation under the
% independent normal and inverse-gamma prior; constructs the one-step-ahead
% posterior predictive density for 2020Q1 by averaging conditional Gaussian
% forecasts across posterior draws.

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

% one-step-ahead regressor vector (T+1)
xTp1 = [1; y(end); y(end-1)];
y_obs = 1.39;
ygrid = linspace(-3,8,300)';

% initialize chain
beta = (X'*X)\(X'*y);
e    = y - X*beta;
sig2 = (e'*e)/T;

store_theta = zeros(nsim, k+1);     % [beta' sig2]
store_fy = zeros(size(ygrid)); % store only the sum

% Gibbs sampler
for isim = 1:nsim + burnin
    % sample beta
    Dbeta = (iVbeta + X'*X/sig2)\speye(k); 
    beta_hat = Dbeta*(iVbeta*beta0 + X'*y/sig2);
    C = chol(Dbeta,'lower');
    beta = beta_hat + C*randn(k,1);

    % sample sig2
    e = y - X*beta;
    sig2 = 1/gamrnd(nu0 + T/2, 1/(S0 + 0.5*(e'*e)));

    if isim > burnin
        isave = isim - burnin;
        store_theta(isave,:) = [beta' sig2];

        % store one-step-ahead predictive density
        mu1 = xTp1'*beta;
        s2_1 = sig2;                           
        fy = normpdf(ygrid, mu1, sqrt(s2_1));       
        store_fy = store_fy + fy;
    end
end

% posterior summaries
theta_mean = mean(store_theta,1);
theta_CI  = quantile(store_theta,[.025 .975],1);

disp('Posterior mean of [beta'' sig2]:');
disp(theta_mean);
disp('95% posterior intervals (rows: 2.5%, 97.5%):');
disp(theta_CI);

% Plot predictive density
fy_hat = store_fy/nsim;
figure;
hold on;
plot(ygrid, fy_hat, 'k', 'LineWidth', 2);
xline(y_obs, 'k--', 'LineWidth', 2);
hold off; box off;
xlabel('$y_{T+1}$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$p(y_{T+1}\mid\mathbf{y},\mathbf{x}_{T+1})$', ...
       'Interpreter', 'latex', 'FontSize', 14);
