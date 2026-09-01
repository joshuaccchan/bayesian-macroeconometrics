% UC_AR.m
% Gibbs sampler for a local level (unobserved components) model of US CPI
% inflation with an AR(1) cyclical component. The model is
%   y_t   = tau_t + eps_t,
%   eps_t = rho*eps_{t-1} + u_t,    u_t   ~ N(0, sig2),
%   tau_t = tau_{t-1} + eta_t,      eta_t ~ N(0, omega2),
% with priors tau0 ~ N(a0, b0), rho ~ U(-1, 1), and inverse-gamma priors on
% sig2 and omega2. The trend tau is drawn in one block from its Gaussian full
% conditional using a precision (band-matrix) sampler; rho is drawn from a
% truncated normal via tnormrnd.m; and sig2, omega2, and tau0 from standard
% conjugate updates.
%
% Requires: tnormrnd.m

clear; clc; rng(42);
nsim = 50000; burnin = 1000;
% load data - US CPI 1948M1 - 2019M12
data = readmatrix('USCPI.csv', 'Range', 'B2:B865');
y = data; 
T = size(y,1);

% initialize for storage
store_tau = zeros(nsim,T);
store_theta = zeros(nsim,4);  % [rho,sig2,omega2,tau0]

% prior hyperparameters
a0 = 5; b0 = 100;
nu_sig0 = 3; S_sig0 = 1*(nu_sig0-1);
nu_omega0 = 3; S_omega0 = .25^2*(nu_omega0-1);

% initialize the Markov chain
sig2 = 1; omega2 = .1; tau0 = 5; rho = 0;
    
% compute a few things outside the loop
S1 = sparse(2:T,1:T-1,1,T,T); % first-lag shift matrix
H = speye(T) - S1;
H_rho = speye(T) - rho*S1;
HH_rho = H_rho'*H_rho;
HH = H'*H;
HHiota = HH*ones(T,1);

for isim = 1:nsim+burnin
    % sample tau
    Ktau = HH/omega2 + HH_rho/sig2;    
    tau_hat = Ktau\(tau0/omega2*HHiota + HH_rho*y/sig2);
    Ctau = chol(Ktau,'lower');
    tau = tau_hat + (Ctau')\randn(T,1); 
    
    % sample rho
    e = y - tau;
    Krho = sum(e(1:T-1).^2)/sig2;
    rho_hat = e(1:T-1)'*e(2:T)/sum(e(1:T-1).^2);
    rho = tnormrnd(rho_hat,1/Krho,-1,1);
    H_rho = speye(T) - rho*S1;
    HH_rho = H_rho'*H_rho;
   
    % sample sig2   
    u = H_rho*e;
    sig2 = 1/gamrnd(nu_sig0 + T/2,1/(S_sig0 + u'*u/2));
    
    % sample omega2        
    omega2 = 1/gamrnd(nu_omega0 + T/2, ...
        1/(S_omega0 + (tau-tau0)'*HH*(tau-tau0)/2));    
    
    % sample tau0
    Ktau0 = 1/b0 + 1/omega2;
    tau0_hat = Ktau0\(a0/b0 + tau(1)/omega2);
    tau0 = tau0_hat + sqrt(1/Ktau0)*randn;
    
    if isim>burnin
        isave = isim-burnin;
        store_tau(isave,:) = tau';      
        store_theta(isave,:) = [rho sig2 omega2 tau0];
    end    
end
tau_mean = mean(store_tau,1)';      
theta_mean = mean(store_theta,1)'; 
tau_q    = quantile(store_tau,[0.05 0.95],1); 
tau_lo   = tau_q(1,:)';
tau_hi   = tau_q(2,:)';

% monthly time axis: 1948M1 to 2019M12
t0 = datetime(1948,1,1);
tt = t0 + calmonths(0:T-1)';

fig = figure; hold on;

hCI = fill([tt; flipud(tt)], [tau_lo; flipud(tau_hi)], ...
           [0.85 0.85 0.85], 'EdgeColor','none');
hCI.HandleVisibility = 'off';   % exclude from legend

hTrend = plot(tt, tau_mean, 'k',  'LineWidth', 1.5);
hInf   = plot(tt, y,        'k:', 'LineWidth', 1);

box off;
xlim([tt(1)-calmonths(12) tt(end)+calmonths(12)]);
ylim([min([y; tau_lo])-1  max([y; tau_hi])+1]);
xlabel('Time','FontSize',14);
ylabel('Inflation','FontSize',14);

lgd = legend([hTrend hInf], {'Trend','Inflation'}, ...
             'Location','best');
lgd.FontSize = 14;

set(gca,'FontSize',14);      % tick labels
set(gcf,'Position',[100 100 900 350]);

% tight layout
set(gca, 'LooseInset', max(get(gca,'TightInset'), 0.02));

% save as EPS with embedded fonts
print(fig, 'CPI_trend', '-depsc2', '-painters');
