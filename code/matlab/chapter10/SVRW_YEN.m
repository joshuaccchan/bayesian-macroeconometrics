% SVRW_YEN.m
% Collapsed Gibbs sampler for the standard (random-walk) stochastic volatility
% model, fitted to daily YEN/USD returns. The model is
%   y_t = exp(h_t/2)*eps_t,   eps_t ~ N(0, 1),
%   h_t = h_{t-1} + u_t,      u_t   ~ N(0, sigh2),
% with priors h0 ~ N(a0, b0) and sigh2 ~ IG(nu_h, S_h). 
%
% Requires: SVRW.m, SV_RW_gaussian_approx.m

clear; clc;
rng(42); % for reproducibility
nsim = 20000; burnin = 1000;
data = readmatrix('YENUSD.csv');
y = data; T = length(y);

% prior hyperparameters
a0 = 0; b0 = 100;
nu_h = 3; S_h = .2^2*(nu_h-1);

% storage
store_theta = zeros(nsim,2); % [h0 sigh2]
store_h = zeros(nsim,T);

% precompute a few things
S1 = sparse(2:T, 1:(T-1), 1, T, T);
H = speye(T) - S1;
HH = H'*H;
c = 1e-4; ystar = log(y.^2 + c);

% initialize
sigh2 = .05;
h0 = log(var(y));
h = SV_RW_gaussian_approx(y.^2,h0,sigh2);
for isim = 1:nsim + burnin    
    % sample sigh2        
    sigh2 = 1/gamrnd(nu_h + T/2, 1/(S_h + (h-h0)'*HH*(h-h0)/2));
    
    % sample h0
    Kh0 = 1/b0 + 1/sigh2;
    h0_hat = Kh0\(a0/b0 + h(1)/sigh2);
    h0 = h0_hat + 1/sqrt(Kh0)*randn;

    % sample h    
    h = SVRW(ystar,h,h0,sigh2);

    if isim > burnin
        isave = isim - burnin;
        store_h(isave,:) = h'; 
        store_theta(isave,:) = [h0, sigh2];
    end    
end
h_std = exp(store_h/2); % transform to standard deviation
h_mean = mean(h_std)'; 
h_CI = quantile(h_std, [0.05 0.95]);
h_lower = h_CI(1,:)';
h_upper = h_CI(2,:)';

theta_mean = mean(store_theta);
theta_CI = quantile(store_theta,[.05 .95]);


tt = linspace(2005,2013,T)';
fig1 = figure; hold on;
plot(tt, y, 'k', 'LineWidth', 1.5);
xlim([2005 2013]);
box off;
xlabel('Time','FontSize',14);
ylim([-6 6]);  
yticks(-6:2:6);

set(gca,'FontSize',14);
set(gca,'TickDir','out');
set(gca,'LineWidth',1.2);
set(gcf,'Position',[100 100 900 350]);
set(gca, 'LooseInset', max(get(gca,'TightInset'), 0.02));
print(fig1, 'YENdata', '-depsc2', '-painters');

fig2 = figure; hold on;
fill([tt; flipud(tt)], ...
     [h_lower; flipud(h_upper)], ...
     [0.8 0.8 0.8], ...
     'EdgeColor','none');
plot(tt, h_mean, 'k', 'LineWidth', 1.5);
xlim([2005 2013]);
box off;
xlabel('Time','FontSize',14);
set(gca,'FontSize',14);
set(gca,'TickDir','out');
set(gca,'LineWidth',1.2);

set(gcf,'Position',[100 100 900 350]);
set(gca, 'LooseInset', max(get(gca,'TightInset'), 0.02));
print(fig2, 'SV_YEN_h', '-depsc2', '-painters');

