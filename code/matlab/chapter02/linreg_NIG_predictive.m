% linreg_NIG_predictive.m
% Closed-form Bayesian inference for an AR(2) model of US PCE inflation
% under the natural conjugate (normal-inverse-gamma) prior; computes the
% one-step-ahead Student-t posterior predictive density for 2020Q1 and
% reports the predictive percentile at the realized value.

clear; clc;

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

% prior hyperparameters (normal-inverse-gamma)
beta0  = zeros(k,1);   % prior mean
iVbeta = speye(k)/100; % prior precision
nu0    = 4;            % prior degrees of freedom
S0     = 1;            % prior scale
    
% Posterior hyperparameters
Dbeta = (iVbeta + X'*X)\speye(k);
beta_hat = Dbeta*(iVbeta*beta0 + X'*y);
S_hat = S0 + (y'*y + beta0'*iVbeta*beta0 ...
    - beta_hat'*(Dbeta\beta_hat))/2;
nu_hat = nu0 + T/2;

% Posterior predictive density for y_{T+1}
xTp1  = [1; y(end); y(end-1)]; % regressor vector at T+1
ygrid = linspace(-3,8,500)';   % evaluation grid
fy = tpdfLS(ygrid, xTp1'*beta_hat, ...
    S_hat/nu_hat*(1 + xTp1'*Dbeta*xTp1),2*nu_hat);

% Plot predictive density
figure;
hold on;
plot(ygrid, fy, 'k', 'LineWidth', 2); 
xline(1.39, 'k--', 'LineWidth', 2); 
hold off; box off;
xlabel('$y_{T+1}$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$p(y_{T+1}\mid\mathbf{y},\mathbf{x}_{T+1})$', ...
       'Interpreter', 'latex', 'FontSize', 14);


y_obs = 1.39;
z = (y_obs - xTp1'*beta_hat) / sqrt(S_hat/nu_hat*(1 + xTp1'*Dbeta*xTp1));
pct = tcdf(z, 2*nu_hat);
fprintf('Predictive percentile at y_{T+1} = 1.39: %.2f%%\n', 100*pct);

