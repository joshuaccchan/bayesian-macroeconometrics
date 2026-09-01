% VAR_NCP.m
% Forecasting with a VAR(p) under the natural conjugate (normal-inverse-
% Wishart) prior, applied to a four-variable quarterly macro panel. With the
% lag order fixed at p, the script reports the log marginal likelihood and
% runs a recursive (expanding-window) one-step-ahead forecasting exercise for
% PCE inflation from 2000Q1 to 2019Q4, reporting the RMSE and plotting the
% forecasts against the realized series. The computations are closed form, so
% the script is deterministic.
%
% Requires: Minn_NCP.m, estimate_VAR_NCP.m, ml_VAR_NCP.m, mgammaln.m, ldet.m

clear; clc;
rng(42); % for reproducibility
p = 7;
kappa1 = 100; kappa2 = 0.04; rw = 0;

% load data
data = readmatrix('macro4_Q.csv', 'NumHeaderLines', 1);
data = data(:,2:end); % drop date column
Y0 = data(1:8,:);  % initial conditions: 1960Q1-1961Q4
Y = data(9:240,:); % 1962Q1-2019Q4
[T,n] = size(Y);
k = n*p+1;

% construct prior using full sample (for marginal likelihood)
[A0, VA, nu0, S0] = Minn_NCP(Y, Y0, p, kappa1, kappa2, rw);
[~,KA,~,S_hat] = estimate_VAR_NCP(Y, Y0, p, A0, VA, nu0, S0);

fprintf('Log ML (macro4_Q, p=%d): %.2f\n', p, ml_VAR_NCP(VA,S0,nu0,KA,S_hat,T));

% recursive forecasting from 2000Q1 to 2019Q4
% 2000Q1 corresponds to row (2000-1962)*4+1 = 153
t0 = (2000-1962)*4 + 1;
Yfull = [Y0; Y]; % stack pre-sample with sample
var_idx = 2;     % PCE inflation
nFcst = T - t0 + 1;
fcst = zeros(nFcst,1);
actual = Y(t0:T, var_idx);
fcstDates = (2000:.25:2000+.25*(nFcst-1))';

for t = t0:T
    Yt  = Y(1:t-1, :);  % estimation sample up to time t-1
    Y0t = Y0;
    % construct prior using data up to time t-1
    [A0t, VAt, nu0t, S0t] = Minn_NCP(Yt, Y0t, p, kappa1, kappa2, rw);
    A_hat = estimate_VAR_NCP(Yt, Y0t, p, A0t, VAt, nu0t, S0t);
    r = size(Y0,1) + t;
        % build x_t = (1, y_{t-1}', ..., y_{t-p}')
    xt = [1; reshape(Yfull(r-1:-1:r-p, :)', [], 1)];
    yhat = xt' * A_hat;
    fcst(t - t0 + 1) = yhat(var_idx);
end
err  = actual - fcst;
RMSE = sqrt(mean(err.^2));

fprintf('1-step PCE inflation forecast (2000Q1-2019Q4): RMSE = %.3f\n', RMSE);

fig1 = figure; hold on;
p1 = plot(fcstDates, actual, 'k-',  'LineWidth', 1.8); % actual
p2 = plot(fcstDates, fcst,   'k--', 'LineWidth', 1.5); % forecast

xlim([fcstDates(1) fcstDates(end)]);
box off;
xlabel('Time','FontSize',14);
ylabel('PCE inflation','FontSize',14);

legend([p1 p2], {'Actual','Forecast'}, ...
       'FontSize',12, 'Location','best');

set(gca,'FontSize',14);
set(gca,'TickDir','out');
set(gca,'LineWidth',1.2);

set(gcf,'Position',[100 100 900 350]);
set(gca,'LooseInset',max(get(gca,'TightInset'),0.02));

print(fig1,'VAR_forecast','-depsc2','-painters');
