% SVM_SP500_HMC.m
% Gibbs sampler with an HMC update of the log-volatility for the stochastic
% volatility in mean (SVM) model, fitted to daily S&P 500 excess returns. The
% model is
%   y_t = mu + alp*exp(h_t) + eps_t,   eps_t ~ N(0, exp(h_t)),
%   h_t = h_{t-1} + u_t,               u_t   ~ N(0, sigh2),
% with priors gam = (mu, alp)' ~ N(gam0, iVgam^{-1}), sigh2 ~ IG(nu_h, S_h),
% and h0 ~ N(a0, b0). The regression coefficients, sigh2, and h0 use conjugate
% updates; the log-volatility h is sampled by Hamiltonian Monte Carlo using the
% local functions logpost_svm_h, grad_logpost_svm_h, and leapfrog below.
%
% Requires: SV_RW_gaussian_approx.m

clear; clc;
rng(42); % for reproducibility
nsim = 20000; burnin = 1000;

% prepare the data
sp500_raw = readmatrix('SP500.csv'); % [date, index level]
dff_raw   = readmatrix('DFF.csv'); % [date, fed funds rate]
    % keep only valid S&P 500 obs.
valid_idx = sp500_raw(:,2) ~= 0;
sp500_raw = sp500_raw(valid_idx,:);
sp500_dates   = sp500_raw(2:end,1);
sp500_returns = 100 * log(sp500_raw(2:end,2) ...
    ./ sp500_raw(1:end-1,2));
    % match return date with the corresponding DFF obs.
[is_matched, dff_loc] = ismember(sp500_dates, ...
    dff_raw(:,1));
sp500_returns = sp500_returns(is_matched);
dff_loc       = dff_loc(is_matched);
    % annualized fed funds rate (percent) to daily
y = sp500_returns - dff_raw(dff_loc,2)/252;
T = length(y);

% prior hyperparameters
gam0 = zeros(2,1); iVgam = speye(2)/100;
nu_h = 3; S_h = .2^2*(nu_h-1);
a0 = 0; b0 = 100;

% HMC settings for h
eps = 0.04; % step size
L = 20;     % # of leapfrog steps
accept_h = 0; % acceptance counter

% precompute banded prior precision pieces
S1 = sparse(2:T, 1:(T-1), 1, T, T);
H  = speye(T) - S1;
HH = H'*H;

% initialize
alp  = 0;
mu = mean(y);
h0   = log(var(y));
sigh2 = .2;
    % initialize h using Gaussian approximation
h = SV_RW_gaussian_approx((y - mu).^2, h0, sigh2);

% storage
store_theta = zeros(nsim,3); % [mu alp sigh2]
store_h = zeros(nsim,T);
for isim = 1:(nsim+burnin)
    % sample gam = (mu, alp)'    
    X = [ones(T,1) exp(h)];
    iSy = sparse(1:T,1:T,1./exp(h));
    XiSy = X'*iSy;
    Kgam = iVgam + XiSy*X;
    gam_hat = Kgam\(iVgam*gam0 + XiSy*y);
    gam = gam_hat + chol(Kgam,'lower')'\randn(2,1);
    mu = gam(1); alp = gam(2);

    % sample h using HMC
    logpost = @(ht) logpost_svm_h(y, mu, alp, ht,...
        h0, sigh2, HH);
    grad = @(ht) grad_logpost_svm_h(y, mu, alp, ht, ...
        h0, sigh2, HH);
    p0 = randn(T,1); % initial momentum
    [hc, pc] = leapfrog(h, p0, eps, L, grad);
    H0 = -logpost(h)  + 0.5*(p0'*p0);
    Hc = -logpost(hc) + 0.5*(pc'*pc);
    if log(rand) < -(Hc - H0)
        h = hc;
        if isim > burnin
            accept_h = accept_h + 1;
        end
    end

    % % sample h using a Laplace-based ARMH step
    % [h, accept] = sample_SVM_h_ARMH(y, alp, mu, h, h0, sigh2, HH);
    % if isim > burnin
    %     accept_h = accept_h + accept;
    % end
    
    % sample sigh2
    sigh2 = 1/gamrnd(nu_h + T/2, ...
        1/(S_h + (h-h0)'*HH*(h-h0)/2));

    % sample h0    
    Kh0 = 1/b0 + 1/sigh2;
    h0_hat = (a0/b0 + h(1)/sigh2) / Kh0;
    h0 = h0_hat + randn/sqrt(Kh0);

    if isim > burnin
        isave = isim - burnin;
        store_h(isave,:)     = h';
        store_theta(isave,:) = [mu alp sigh2];
    end
end
fprintf('HMC acceptance rate = %.3f\n', accept_h/nsim);
% posterior summaries
h_mean = mean(exp(store_h/2), 1)'; 
h_CI = quantile(exp(store_h/2), [0.05 0.95], 1); % 90% credible interval
h_lower = h_CI(1,:)';
h_upper = h_CI(2,:)';

theta_mean = mean(store_theta);
theta_CI  = quantile(store_theta, [0.025 0.975]); % 95% credible interval

% Plot posterior mean and 90% credible interval for exp(h_t/2)
tt = linspace(2013, 2016, T)';

fig = figure;
hold on;
fill([tt; flipud(tt)], [h_lower; flipud(h_upper)], ...
     [0.8 0.8 0.8], 'EdgeColor','none');
plot(tt, h_mean, 'k', 'LineWidth', 1.5); 
hold off; box off;

xlim([2013 2016]);
xticks([2013 2014 2015 2016]);   % <-- only these ticks
set(gca,'FontSize',14);

set(gcf,'Position',[100 100 900 350]);
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));

print(fig, 'SVM_h', '-depsc2', '-painters');

function lden = logpost_svm_h(y, mu, alp, h, h0, sigh2, HH)
% Log conditional posterior density of the log-volatility h (up to an additive
% constant) in the SVM model, combining the Gaussian likelihood with the
% random-walk prior. 
r = y - mu - alp*exp(h);
lden = -0.5*sum(h) - 0.5*sum((r.^2).*exp(-h)) ...
    - 0.5/sigh2 * (h - h0)' * HH * (h - h0);
end

function grad = grad_logpost_svm_h(y, mu, alp, h, h0, sigh2, HH)
% Gradient of logpost_svm_h with respect to h (same inputs).
grad_like = -0.5 - 0.5*alp^2*exp(h) ...
    + 0.5*(y - mu).^2.*exp(-h);
grad_prior = -(HH*(h - h0))/sigh2;
grad = grad_like + grad_prior;
end

function [hNew, pNew] = leapfrog(h, p, eps, L, grad_logpost)
% Leapfrog integrator for HMC: L steps of size eps evolving (h, p) along the
% Hamiltonian trajectory, with adjacent half-steps combined (L+1 gradient
% evaluations). grad_logpost is a function handle returning grad log p(h).
hNew = h;
pNew = p;
pNew = pNew + 0.5*eps*grad_logpost(hNew);
for i = 1:L
    hNew = hNew + eps*pNew;
    if i < L
        pNew = pNew + eps*grad_logpost(hNew);
    else
        pNew = pNew + 0.5*eps*grad_logpost(hNew);
    end
end
pNew = -pNew;
end