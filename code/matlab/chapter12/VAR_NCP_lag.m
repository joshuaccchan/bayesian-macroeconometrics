% VAR_NCP_lag.m
% Selects the lag order of the VAR under the natural conjugate prior by
% computing the log marginal likelihood for p = 1,...,8 on macro4_Q, holding
% the estimation sample fixed across lag lengths. The computations are
% closed form, so the script is deterministic.
%
% Requires: Minn_NCP.m, estimate_VAR_NCP.m, ml_VAR_NCP.m, mgammaln.m, ldet.m

clear; clc;
kappa1 = 100; kappa2 = 0.04; rw = 0;

% load data
data = readmatrix('macro4_Q.csv', 'NumHeaderLines', 1);
data = data(:,2:end); % drop date column
Y0 = data(1:8,:);  % initial conditions: 1960Q1-1961Q4
Y = data(9:240,:); % 1962Q1-2019Q4
T = size(Y,1);

% log marginal likelihood for each lag length
lml = zeros(8,1);
for p = 1:8
    [A0, VA, nu0, S0] = Minn_NCP(Y, Y0, p, kappa1, kappa2, rw);
    [~,KA,~,S_hat] = estimate_VAR_NCP(Y, Y0, p, A0, VA, nu0, S0);
    lml(p) = ml_VAR_NCP(VA, S0, nu0, KA, S_hat, T);
    fprintf('log ML (p = %d): %.1f\n', p, lml(p));
end
