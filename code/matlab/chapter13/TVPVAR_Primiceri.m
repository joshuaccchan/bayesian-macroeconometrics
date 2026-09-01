% TVPVAR_Primiceri.m
% Application of the TVP-VAR with stochastic volatility to U.S. GDP
% deflator inflation, unemployment, and the 3-month T-bill rate, replicating
% the dataset and sample of Primiceri (2005).
%
% Requires: estimate_TVPVAR.m, construct_IR.m, plotCI.m, SURform.m, SVRW.m

clear; clc; rng(42);

p      = 2;
nsim   = 20000;
burnin = 5000;
n_hz   = 21;  % impulse response horizon: 0 to 20 quarters

% load data: 1953Q1-2001Q3, matching Primiceri's sample.
% macro_Primiceri_Q.csv is built by build_macro_Primiceri.m from FRED
% series GDPDEF, UNRATE, TB3MS.
data  = readmatrix('macro_Primiceri_Q.csv', 'NumHeaderLines', 1, ...
                   'Range', 'B:D');
Y_all = data(1:195, :);                  % 1953Q1-2001Q3
[Tall, n] = size(Y_all);
k  = 1 + n*p;
nk = n*k;
m  = n*(n-1)/2;

% split: first 40 obs for prior calibration, rest for inference
T_train = 40;
Y_train = Y_all(1:T_train, :);           % 1953Q1-1962Q4
Y_est   = Y_all(T_train+1-p : end, :);   % 1962Q3-2001Q3 (carries p lags)
T_eff   = Tall - T_train;                % 155 effective obs (1963Q1-2001Q3)
fprintf('Sample: %d obs total, %d for training, T_eff = %d.\n', ...
    Tall, T_train, T_eff);

% indices of free entries below diagonal of L, eq-by-eq order
L_id = nonzeros(tril(reshape(1:n^2, n, n), -1)');

% training-sample OLS for prior centers and scales
T_tr_eff   = T_train - p;
X_tilde_tr = zeros(T_tr_eff, n*p);
for i = 1:p
    X_tilde_tr(:, (i-1)*n+1 : i*n) = Y_train(p-i+1 : end-i, :);
end
Z_tr     = [ones(T_tr_eff, 1) X_tilde_tr];
A_tr     = Z_tr \ Y_train(p+1:end, :);
E_tr     = Y_train(p+1:end, :) - Z_tr * A_tr;
Sig_tr   = (E_tr' * E_tr) / T_tr_eff;
beta_ols = reshape(A_tr', nk, 1);
Vb_ols   = kron(Sig_tr, (Z_tr' * Z_tr) \ eye(k));

% l_OLS from modified-Cholesky Sig_tr = L^{-1} D (L')^{-1}
Lc_tr = chol(Sig_tr, 'lower');
L_ols = diag(diag(Lc_tr)) / Lc_tr;
L_ols = tril(L_ols);
l_ols = L_ols(L_id);

% per-equation sampling-variance blocks for l_OLS
sig2_full = diag(Lc_tr).^2;
Vl_blocks = cell(n-1, 1);
for ii = 1:n-1
    Ein           = -E_tr(:, 1:ii);
    Vl_blocks{ii} = sig2_full(ii+1) * ((Ein' * Ein) \ eye(ii));
end
Vl_full = zeros(m, m); off = 0;
for ii = 1:n-1
    Vl_full(off+1:off+ii, off+1:off+ii) = Vl_blocks{ii};
    off = off + ii;
end

% Prior calibration:
kQ = 0.01; kS = 0.1;
prior = struct();
prior.beta0_mean = beta_ols;
prior.beta0_var  = 100 * eye(nk);     % diffuse prior on beta_0
prior.l0_mean    = l_ols;
prior.l0_var     = 100 * eye(m);      % diffuse prior on l_0
prior.h0_mean    = log(diag(Lc_tr).^2);
prior.h0_var     = 100 * eye(n);      % diffuse prior on h_0
prior.Q_nu       = T_train;
prior.Q_S0       = kQ^2 * T_train * Vb_ols;
prior.S_nu       = (1:n-1) + 1;
prior.S_S0       = cell(n-1, 1);
for ii = 1:n-1
    prior.S_S0{ii} = kS^2 * (ii+1) * Vl_blocks{ii};
end
prior.sigh2_nu = 1 * ones(n, 1);
prior.sigh2_S0 = 0.5 * ones(n, 1);

% run the Gibbs sampler
fprintf('Running Gibbs sampler for TVP-VAR (nsim=%d, burnin=%d) ...\n', nsim, burnin);
tic;
[store_beta, store_l, store_h, ~, store_Q] = estimate_TVPVAR(Y_est, p, prior, nsim, burnin);
toc;
Q_mean = reshape(mean(store_Q, 1), nk, nk);
fprintf('\nQ posterior mean diag stats: min=%.3e median=%.3e max=%.3e\n', ...
    min(diag(Q_mean)), median(diag(Q_mean)), max(diag(Q_mean)));

dates_q = (1963 : 0.25 : 1963 + 0.25*(T_eff-1))';

% residual standard deviations: sqrt(diag(Sigma_t)) at each draw
l_3d = reshape(store_l, nsim, m, T_eff);
l21  = squeeze(l_3d(:, 1, :));
l31  = squeeze(l_3d(:, 2, :));
l32  = squeeze(l_3d(:, 3, :));
d1   = exp(store_h(:, :, 1));
d2   = exp(store_h(:, :, 2));
d3   = exp(store_h(:, :, 3));

sig2_red = zeros(nsim, T_eff, n);
sig2_red(:, :, 1) = d1;
sig2_red(:, :, 2) = l21.^2 .* d1 + d2;
sig2_red(:, :, 3) = (l32 .* l21 - l31).^2 .* d1 + l32.^2 .* d2 + d3;
sig_red = sqrt(sig2_red);

sig_med = squeeze(quantile(sig_red, 0.50, 1));
sig_lo  = squeeze(quantile(sig_red, 0.16, 1));
sig_hi  = squeeze(quantile(sig_red, 0.84, 1));

% Figure 1: residual standard deviations over time
resnames = {'Inflation', 'Unemployment', '3-month T-bill'};
figure;
for ii = 1:n
    subplot(n, 1, ii);
    hold on
        plotCI(dates_q, sig_lo(:, ii), sig_hi(:, ii));
        plot(dates_q, sig_med(:, ii), 'k', 'LineWidth', 1.5);
    hold off
    box off;
    xlim([dates_q(1) dates_q(end)]);
    if ii == n; xlabel('Year'); end
end
set(gcf, 'Position', [100 100 700 500]);
print(gcf, 'Primiceri_sigma', '-depsc');

% MP shock IRFs at three reference dates
ref_dates  = [1975, 1981.5, 1996];       % 1975Q1, 1981Q3, 1996Q1
ref_labels = {'1975Q1', '1981Q3', '1996Q1'};
n_ref      = numel(ref_dates);
i_mp       = n;                          % MP shock = last equation
shock      = zeros(n, 1); shock(i_mp) = 1;
F_sub      = [eye((p-1)*n), zeros((p-1)*n, n)];

store_yIR = zeros(nsim, n_hz, n, n_ref);
maxeig_all = zeros(nsim, n_ref);
for r = 1:n_ref
    [~, t_idx] = min(abs(dates_q - ref_dates(r)));
    for s = 1:nsim
        beta_path = reshape(store_beta(s, :), nk, T_eff);
        beta_t    = beta_path(:, t_idx);
        A_full    = reshape(beta_t, k, n)';
        F         = [A_full(:, 2:end); F_sub];
        maxeig_all(s, r) = max(abs(eig(F)));
        l_path = reshape(store_l(s, :), m, T_eff);
        l_t    = l_path(:, t_idx);
        h_t    = squeeze(store_h(s, t_idx, :));
        Lt = eye(n); Lt(L_id) = l_t;
        Dt = diag(exp(h_t));
        Sig_t = Lt \ Dt / Lt';
        store_yIR(s, :, :, r) = construct_IR(beta_t, Sig_t, n_hz, shock);
    end
    fprintf('Reference %s -> t = %3d : max|eig(F)| median=%.3f  q10=%.3f  q90=%.3f  max=%.3f\n', ...
        ref_labels{r}, t_idx, ...
        quantile(maxeig_all(:, r), 0.5), ...
        quantile(maxeig_all(:, r), 0.1), ...
        quantile(maxeig_all(:, r), 0.9), ...
        max(maxeig_all(:, r)));
end

% normalize by posterior-median impact FFR response (one scalar per ref date)
% so the IRF corresponds to a 1pp shock; dividing each draw by its own
% impact response amplifies noise from draws with near-zero impact
for r = 1:n_ref
    impact_med = quantile(store_yIR(:, 1, i_mp, r), 0.50);
    store_yIR(:, :, :, r) = store_yIR(:, :, :, r) / impact_med;
end

yIR_med = squeeze(quantile(store_yIR, 0.50));
yIR_lo  = squeeze(quantile(store_yIR, 0.16));
yIR_hi  = squeeze(quantile(store_yIR, 0.84));

% Figure 2: inflation and unemployment IRFs, shared y-axis per row
varnames_resp = {'Inflation', 'Unemployment'};
n_resp        = numel(varnames_resp);
hz            = (0:n_hz-1)';

y_lim = zeros(n_resp, 2);
for ii = 1:n_resp
    y_min = min(yIR_lo(:, ii, :), [], 'all');
    y_max = max(yIR_hi(:, ii, :), [], 'all');
    pad   = 0.05 * (y_max - y_min);
    y_lim(ii, :) = [y_min - pad, y_max + pad];
end

figure;
for ii = 1:n_resp
    for r = 1:n_ref
        subplot(n_resp, n_ref, (ii-1)*n_ref + r);
        hold on
            plotCI(hz, yIR_lo(:, ii, r), yIR_hi(:, ii, r));
            plot(hz, yIR_med(:, ii, r), 'k', 'LineWidth', 1.5);
            yline(0, 'k-', 'LineWidth', 0.5);
        hold off
        box off;
        xlim([-0.5 n_hz-1]);
        ylim(y_lim(ii, :));
        if r == 1;       ylabel(varnames_resp{ii}); end
        if ii == n_resp; xlabel('Quarters'); end
    end
end
set(gcf, 'Position', [100 100 800 350]);
print(gcf, 'Primiceri_MPshock_IR', '-depsc');

% save plot data so replot_Primiceri.m can rebuild the figures without rerunning
save('Primiceri_results.mat', 'sig_med', 'sig_lo', 'sig_hi', 'dates_q', ...
     'yIR_med', 'yIR_lo', 'yIR_hi', 'hz', 'ref_labels', 'i_mp', ...
     'n', 'p', 'k', 'nk', 'm', 'T_eff', '-v7');
