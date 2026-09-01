% gpr_uncertainty.m
% Empirical-Bayes Gaussian process regression of quarterly US real GDP growth
% on its own lag and lagged macroeconomic uncertainty. The model is
%   y_t = f(x_t) + eps_t,  eps_t ~ N(0, sig2),  f ~ GP(0, k),
%   x_t = (y_{t-1}, U_{t-1})',
% with an ARD squared exponential kernel
%   k(x,x') = sigf2 * exp(-0.5 * sum_j (x_j - x'_j)^2 / lam_j^2).
% The hyperparameters (sigf2, lam_1, lam_2, sig2) are set by maximizing the
% log integrated likelihood (gpr_fit_eb.m); the fitted mean surface is sliced
% at low/median/high lagged growth and plotted against uncertainty. The
% uncertainty index is from Jurado-Ludvigson-Ng (2015); sample 1960Q4-2019Q4.
%
% Requires: gpr_fit_eb.m, gpr_predict.m, shaded_band.m

clear; clc; rng(42);

% load data; keep the pre-COVID sample (through 2019Q4)
tbl = readtable('gpr_uncertainty_data.csv');
yr = str2double(extractBefore(tbl.date,'Q'));
qt = str2double(extractAfter(tbl.date,'Q'));
tbl = tbl((yr + (qt-1)/4) <= 2019.9, :);

y = tbl.gdp_growth; 
X = [tbl.gdp_growth_lag1, tbl.macro_unc_h1_lag1]; % [lagged growth, lagged unc]
n = size(X,1);

% center response (zero-mean GP) and standardize each input
my = mean(y);  yc = y - my;
mx = mean(X);  sx = std(X);  Xs = (X - mx)./sx;

% empirical-Bayes hyperparameters (maximize the log integrated likelihood)
opts.nstart = 12; opts.verbose = false;
[hp, llf] = gpr_fit_eb(yc, Xs, opts);

nm = {'lagged GDP growth','lagged uncertainty'};
fprintf('Empirical-Bayes ARD hyperparameters (n = %d, %s-%s)\n', ...
        n, tbl.date{1}, tbl.date{end});
fprintf('  signal variance sig_f^2 = %.4f\n', hp.sigf2);
for j = 1:2
    fprintf('  length scale lambda_%d = %6.3f (std)  = %7.4f (%s units)\n', ...
            j, hp.lam(j), hp.lam(j)*sx(j), nm{j});
end
fprintf('  noise variance sig^2 = %.4f (sig = %.3f)\n', hp.sig2, sqrt(hp.sig2));
fprintf('  log integrated likelihood = %.3f\n', llf);
[~,less] = max(hp.lam);
fprintf('  -> %s has the larger length scale (less relevant locally)\n', nm{less});

% slices: fix lagged growth, vary uncertainty
gvals = quantile(X(:,1), [0.10 0.50 0.90]);  % low / median / high growth
glab  = {'10th pct','median','90th pct'};
ug    = linspace(min(X(:,2)), max(X(:,2)), 300)'; % uncertainty grid

z95   = 1.96;
sty   = {':','-','--'};                      % low / median / high
figure('Units','points','Position',[100 100 660 260]);
set(gcf,'PaperPositionMode','auto'); hold on;

% 95% credible band for the median slice
[mum, s2m] = gpr_predict(yc, Xs, ([gvals(2)*ones(size(ug)), ug]-mx)./sx, hp);
sdm = sqrt(s2m);
shaded_band(ug, mum+my-z95*sdm, mum+my+z95*sdm, 0.85);

hm = gobjects(3,1);
for s = 1:3   
    [mu] = gpr_predict(yc, Xs, ([gvals(s)*ones(size(ug)), ug]-mx)./sx, hp);
    hm(s) = plot(ug, mu+my, ['k' sty{s}], 'LineWidth', 2);
end
hold off; box off; set(gca,'FontSize',15); xlim([min(X(:,2)) max(X(:,2))]);
xlabel('Macro uncertainty','FontSize',15);
ylabel('Real GDP growth','FontSize',15);
lgd = legend(hm, ...
   {sprintf('$y_{t-1}=%.1f\\%%$ (10\\%%-tile)', gvals(1)), ...
    sprintf('$y_{t-1}=%.1f\\%%$ (50\\%%-tile)', gvals(2)), ...
    sprintf('$y_{t-1}=%.1f\\%%$ (90\\%%-tile)', gvals(3))}, ...
    'Location','southwest','Interpreter','latex','FontSize',13);
lgd.Box = 'off';

% export figure
print(gcf,'-depsc2','gpr_uncertainty.eps');

