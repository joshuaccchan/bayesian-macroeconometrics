% UCSV.m
% Gibbs sampler for the unobserved components model with stochastic volatility
% of Stock and Watson (2007), fitted to quarterly PCE inflation. The
% model is
%   y_t   = tau_t + eps^y_t,         eps^y_t   ~ N(0, exp(h_t)),
%   tau_t = tau_{t-1} + eps^tau_t,   eps^tau_t ~ N(0, exp(g_t)),
% with random-walk log-volatilities h_t (gap) and g_t (trend), each with an
% inverse-gamma prior on its innovation variance. 
%
% Requires: SVRW.m, SV_RW_gaussian_approx.m

clear; clc;
rng(42); % for reproducibility
nsim   = 50000; burnin = 1000;

% load PCE data - 1960Q1-2024Q4
data = readmatrix('USPCE.csv', 'Range', 'B2:B261'); 
y = data;
T = size(y,1);

% prior hyperparameters
a0_h = 0; b0_h = 10; % h0 ~ N(a0_h, b0_h)
a0_g = 0; b0_g = 10; % g0 ~ N(a0_g, b0_g)
a0_tau = 0;  b0_tau = 10; % tau0 ~ N(a0_tau, b0_tau)    
nu_oh = 3; S_oh = 0.2^2*(nu_oh-1); % omega_h^2 ~ IG(nu_oh, S_oh)
nu_og = 3; S_og = 0.2^2*(nu_og-1); % omega_g^2 ~ IG(nu_og, S_og)

% precompute a few things 
c = 1e-4; % log-squared safeguard
S1 = sparse(2:T, 1:(T-1), 1, T, T);
H  = speye(T) - S1;
HH = H'*H;

% initialize
tau0 = mean(y);
tau  = tau0*ones(T,1);
h0 = log(var(y)); g0 = log(var(y));
omega_h2 = 0.1;   omega_g2 = 0.1; 
    % initialize h using Gaussian approximation
h = SV_RW_gaussian_approx((y - tau).^2, h0, omega_h2);
    % initialize g using Gaussian approximation
dtau = tau - [tau0; tau(1:end-1)];
g = SV_RW_gaussian_approx(dtau.^2, g0, omega_g2);

% storage
    %[omega_h2 omega_g2 h0 g0 tau0]
store_theta = zeros(nsim,5);  
store_tau = zeros(nsim,T);
store_h = zeros(nsim,T);
store_g = zeros(nsim,T);

for isim = 1:(nsim+burnin)

    % sample tau    
    iOh = sparse(1:T,1:T,exp(-h)); % Omega_h^{-1}
    HiOgH = H'*sparse(1:T,1:T,exp(-g))*H; 
    Ktau = HiOgH + iOh;
    tau_mean = Ktau\(HiOgH*(tau0*ones(T,1)) + iOh*y);
    tau = tau_mean + chol(Ktau,'lower')'\randn(T,1);

    % sample tau0    
    Ktau0 = 1/b0_tau + exp(-g(1));
    tau0_hat = (a0_tau/b0_tau + tau(1)*exp(-g(1)))/Ktau0;
    tau0 = tau0_hat + (1/sqrt(Ktau0))*randn;
    
    % sample h     
    ystar_h = log((y - tau).^2 + c);
    h = SVRW(ystar_h, h, h0, omega_h2);

    % sample omega_h2    
    omega_h2 = 1/gamrnd(nu_oh + T/2, ...
        1/(S_oh + 0.5*(h - h0)'*HH*(h - h0)));

    % sample h0    
    Kh0 = 1/b0_h + 1/omega_h2;
    h0_hat = (a0_h/b0_h + h(1)/omega_h2) / Kh0;
    h0 = h0_hat + (1/sqrt(Kh0))*randn;
    
    % sample g    
    dtau = tau - [tau0; tau(1:end-1)];
    ystar_g = log(dtau.^2 + c);
    g = SVRW(ystar_g, g, g0, omega_g2);

    % sample omega_g2    
    omega_g2 = 1/gamrnd(nu_og + T/2, ...
        1/(S_og + 0.5*(g - g0)'*HH*(g - g0)));
    
    % sample g0    
    Kg0 = 1/b0_g + 1/omega_g2;
    g0_hat = (a0_g/b0_g + g(1)/omega_g2)/Kg0;
    g0 = g0_hat + (1/sqrt(Kg0))*randn;

    if isim > burnin
        isave = isim - burnin;
        store_tau(isave,:) = tau';
        store_h(isave,:) = h';
        store_g(isave,:) = g';
        store_theta(isave,:) = [omega_h2 omega_g2 h0 g0 tau0];
    end
end
% posterior summaries
theta_mean = mean(store_theta)';
tau_mean = mean(store_tau)';
h_mean = mean(exp(store_h/2))';
g_mean = mean(exp(store_g/2))';

tt = (1960:.25:2024.75)';
fig1 = figure; hold on;
p1 = plot(tt, y, 'k--', 'LineWidth', 1.2); % data
p2 = plot(tt, tau_mean,'k', 'LineWidth', 1.8); % trend

xlim([1959.75 2025]);
box off;
xlabel('Time','FontSize',14);


legend([p1 p2], {'Actual inflation','Trend inflation'}, ...
       'FontSize',12, 'Location','best');

set(gca,'FontSize',14);
set(gca,'TickDir','out');
set(gca,'LineWidth',1.2);

set(gcf,'Position',[100 100 900 350]);
set(gca,'LooseInset',max(get(gca,'TightInset'),0.02));

print(fig1,'UCSV_trend','-depsc2','-painters');

fig2 = figure; hold on;

p1 = plot(tt, h_mean, 'k',  'LineWidth', 1.8); % gap volatility
p2 = plot(tt, g_mean, 'k--','LineWidth', 1.5); % trend volatility

xlim([1959.75 2025]);
box off;
xlabel('Time','FontSize',14);


legend([p1 p2], {'Gap volatility','Trend volatility'}, ...
       'FontSize',12, 'Location','best');

set(gca,'FontSize',14);
set(gca,'TickDir','out');
set(gca,'LineWidth',1.2);

set(gcf,'Position',[100 100 900 350]);
set(gca,'LooseInset',max(get(gca,'TightInset'),0.02));

print(fig2,['UCSV_SV'],'-depsc2','-painters');