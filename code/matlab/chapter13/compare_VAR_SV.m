% compare_VAR_SV.m
% Recursive one-step-ahead forecasting of PCE inflation across three VAR
% specifications: homoskedastic, Cholesky stochastic volatility, and
% order-invariant stochastic volatility.
%
% Requires: Minn_indep.m, pred_VAR_homo.m, pred_VAR_SV.m, pred_VAR_OISV.m,
%           SURform2.m, SVRW.m, sample_B0.m, SVAR1.m, sample_SVAR1para.m

clear; clc;
rng(42); % for reproducibility
p = 7;
nsim = 10000;
burnin = 1000;
kappa1 = 100;       % intercept prior variance
kappa2 = 0.2^2;     % own-lag tightness
kappa3 = 0.2^2/4;   % cross-lag tightness
rw      = 0;        % 0 = zero prior mean (growth-rate data); 1 = RW
var_idx = 2;        % PCE inflation

% load data
data = readmatrix('macro4_Q.csv', 'NumHeaderLines', 1);
data = data(:, 2:end);   % drop date column
Y0 = data(1:8, :);       % initial conditions: 1960Q1-1961Q4
Y  = data(9:260, :);     % 1962Q1-2024Q4
[T, n] = size(Y);
k = 1 + n*p;

% recursive forecasting from 2000Q1 to 2024Q4
t0     = (2000 - 1962)*4 + 1;
Yfull  = [Y0; Y];
nFcst  = T - t0 + 1;
fcst_homo = zeros(nFcst, 1);
LPL_homo  = zeros(nFcst, 1);
fcst_SV   = zeros(nFcst, 1);
LPL_SV    = zeros(nFcst, 1);
fcst_OISV = zeros(nFcst, 1);
LPL_OISV  = zeros(nFcst, 1);
actual    = Y(t0:T, var_idx);
fcstDates = (2000:.25:2000 + .25*(nFcst-1))';

disp('Recursive one-step-ahead forecasts ...');
tic;
for t = t0:T
    Yt = Y(1:t-1, :);
    Tt = size(Yt, 1);
    yreal = Y(t, var_idx);

    % independent Minnesota prior on beta (data up to t-1)
    [beta0, V_Minn] = Minn_indep(p, kappa1, kappa2, kappa3, Y0, Yt, rw);

    % regressor matrix Z shared across models
    tmpY = [Y0(end-p+1:end, :); Yt];
    Z = ones(Tt, k);
    for i = 1:p
        Z(:, 1 + (i-1)*n + 1 : 1 + i*n) = tmpY(p-i+1:end-i, :);
    end

    % regressor row x_t for forecasting Y(t, :)
    r  = size(Y0, 1) + t;
    xt = [1, reshape(Yfull(r-1:-1:r-p, :)', 1, n*p)];

    % homoskedastic VAR
    [fcst_homo(t-t0+1), LPL_homo(t-t0+1)] = pred_VAR_homo( ...
        Yt, Z, xt, beta0, V_Minn, nsim, burnin, var_idx, yreal);

    % VAR with Cholesky stochastic volatility
    [fcst_SV(t-t0+1), LPL_SV(t-t0+1)] = pred_VAR_SV( ...
        Yt, Z, xt, beta0, V_Minn, nsim, burnin, var_idx, yreal);

    % VAR with order-invariant stochastic volatility
    [fcst_OISV(t-t0+1), LPL_OISV(t-t0+1)] = pred_VAR_OISV( ...
        Yt, Z, xt, beta0, V_Minn, nsim, burnin, var_idx, yreal);

    if mod(t - t0 + 1, 10) == 0
        fprintf('  forecast %d / %d done\n', t - t0 + 1, nFcst);
    end
end
fprintf('Elapsed: %.1f sec\n', toc);

err_homo    = actual - fcst_homo;
RMSE_homo   = sqrt(mean(err_homo.^2));
sumLPL_homo = sum(LPL_homo);
fprintf('Homoskedastic VAR  : RMSE = %.4f   sum LPL = %.4f\n', ...
    RMSE_homo, sumLPL_homo);

err_SV    = actual - fcst_SV;
RMSE_SV   = sqrt(mean(err_SV.^2));
sumLPL_SV = sum(LPL_SV);
fprintf('Cholesky SV VAR    : RMSE = %.4f   sum LPL = %.4f\n', ...
    RMSE_SV, sumLPL_SV);

err_OISV    = actual - fcst_OISV;
RMSE_OISV   = sqrt(mean(err_OISV.^2));
sumLPL_OISV = sum(LPL_OISV);
fprintf('Order-invariant SV : RMSE = %.4f   sum LPL = %.4f\n', ...
    RMSE_OISV, sumLPL_OISV);
