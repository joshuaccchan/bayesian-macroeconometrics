% UC_SVM.m
% Metropolis-within-Gibbs sampler for the unobserved components stochastic
% volatility in mean model with time-varying coefficients, fitted to quarterly
% PCE inflation. The measurement equation is
%   y_t = tau_t + alp_t*exp(h_t) + eps_t,   eps_t ~ N(0, exp(h_t)),
% where the coefficient vector gam_t = (tau_t, alp_t)' follows a random walk and
% the log-volatility h_t is a random walk. The coefficient path gam and the
% static parameters use conjugate updates; h is sampled by the Laplace-based
% acceptance-rejection MH step (sample_SVM_h_ARMH.m). 
% The stacked design matrix is built with SURform.m.
%
% Requires: sample_SVM_h_ARMH.m, SURform.m, SV_RW_gaussian_approx.m

clear; clc;
rng(42); % for reproducibility
nsim = 20000; burnin = 5000;

% load PCE data - 1960Q1-2024Q4
data = readmatrix('USPCE.csv', 'Range', 'B2:B261'); 
y = data;
T = length(y);

% prior hyperparameters
agam = [2,0]'; iVgam = eye(2)/100;
nuOmega = 3;  SOmega = (nuOmega-1)*[0.25^2 0.10^2]';
nuh = 3; Sh  = 0.2^2 * (nuh - 1);
ah = 0; Vh = 100;

% initialize
omega  = [0.25^2, 0.10^2]'; % store the diagonal elements
sigh2  = 0.2^2;
h0 = log(var(y));
tau = mean(y)*ones(T,1);
h = SV_RW_gaussian_approx((y - tau).^2, h0, sigh2);
exp_h = exp(h);
gam0 = zeros(2,1);

% storage    
store_theta = zeros(nsim,6);  % [h0 sigh2 omega' gam0']
store_tau = zeros(nsim,T);
store_alp = zeros(nsim,T);
store_h = zeros(nsim,T);

% precompute fixed matrices
S1gam = sparse(3:2*T,1:2*(T-1),ones(1,2*(T-1)),2*T,2*T);
Hgam = speye(2*T) - S1gam;
S1 = sparse(2:T,1:(T-1),1,T,T);
H  = speye(T) - S1;
HH = H' * H;
accept_h = 0;

for isim = 1:(nsim + burnin)
    % sample gam
    Xgam = SURform([ones(T,1) exp_h]);
    tmp = Xgam' * sparse(1:T,1:T,1./exp_h);
        % prior precision
    Pgam = Hgam' * kron(speye(T), diag(1./omega)) * Hgam; 
    Kgam =  Pgam + tmp * Xgam;
    Cgam = chol(Kgam,'lower');   
    gamhat = Kgam\(Pgam*kron(ones(T,1),gam0) + tmp*y);
    gam = gamhat + Cgam' \ randn(2*T,1);
    tau = gam(1:2:end);
    alp = gam(2:2:end);

    % sample gam0
    Kgam0 = iVgam + diag(1./omega);
    gam0hat = Kgam0 \ (iVgam*agam + gam(1:2)./omega);
    gam0 = gam0hat + chol(Kgam0,'lower')' \ randn(2,1);

    % sample h using acceptance-rejection MH
    [h, accept] = sample_SVM_h_ARMH(y, alp, tau, h, h0, sigh2, HH);
    exp_h = exp(h);
    if isim > burnin
        accept_h = accept_h + accept;
    end

    % sample Omega
    e_gam = reshape(gam - [gam0;gam(1:end-2)],2,T)';
    newSOmega = SOmega + sum(e_gam.^2)'/2;    
    omega = 1./gamrnd(nuOmega + T/2, 1./newSOmega);

    % sample sigh2
    e_h = [h(1)-h0; diff(h)];
    newSh = Sh + sum(e_h.^2)/2;
    sigh2 = 1 / gamrnd(nuh + T/2, 1/newSh);

    % sample h0
    Kh0 = 1/Vh + 1/sigh2;
    h0hat = (ah/Vh + h(1)/sigh2) / Kh0;
    h0 = h0hat + randn / sqrt(Kh0);

    % store draws
    if isim > burnin
        isave = isim - burnin;
        store_tau(isave,:) = tau';
        store_alp(isave,:) = alp';
        store_h(isave,:)   = h';
        store_theta(isave,:) = [h0 sigh2 omega' gam0'];
    end
end
fprintf('Acceptance rate for h = %.3f\n', accept_h/nsim);
% posterior summaries
tau_mean = mean(store_tau, 1)';
tau_CI   = quantile(store_tau, [0.05 0.95], 1);
tau_lower = tau_CI(1,:)';
tau_upper = tau_CI(2,:)';

alp_mean = mean(store_alp, 1)';
alp_CI   = quantile(store_alp, [0.05 0.95], 1);
alp_lower = alp_CI(1,:)';
alp_upper = alp_CI(2,:)';

% plot alpha_t with 90% CI
tt = linspace(1960, 2024.75, T)';
fig = figure;
hold on
fill([tt; flipud(tt)], [alp_lower; flipud(alp_upper)], ...
    [0.8 0.8 0.8], 'EdgeColor', 'none');
plot(tt, alp_mean, 'k', 'LineWidth', 1.5);
hold off
box off
xlim([min(tt)-1 max(tt)+1]);
set(gca, 'FontSize', 14);

set(gcf, 'Position', [100 100 900 300]);
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
print(fig, 'UC_SVM', '-depsc2', '-painters');