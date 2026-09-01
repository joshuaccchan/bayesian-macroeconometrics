% UC_output_gap.m
% Gibbs sampler for a local linear trend (unobserved components) model that
% decomposes 100*log US real GDP into trend (potential) output and a cyclical
% output gap. The model is
%   y_t       = tau_t + c_t,
%   c_t       = phi_1*c_{t-1} + phi_2*c_{t-2} + u_t^c,   u_t^c   ~ N(0, sigc2),
%   D.tau_t   = D.tau_{t-1} + u_t^tau,                   u_t^tau ~ N(0, sigtau2),
% where D denotes the first difference, so trend growth follows a random walk.
% The trend tau is drawn with a precision (band-matrix) sampler; phi by an 
% accept-reject step enforcing stationarity; sigc2 and (tau0, tau_{-1}) 
% from conjugate updates; and sigtau2 with a Griddy-Gibbs step via griddy_gibbs.m.
%
% Requires: griddy_gibbs.m, shade_nber_recessions.m

clear; clc; rng(42);
nsim = 20000; burnin = 1000;
% load data - US GDP 1947Q1 - 2019Q4
data_raw = readmatrix('USGDP.csv', 'Range', 'B2:B293');
data = 100*log(data_raw);
y = data;
T = length(y);

% prior hyperparameters
a0 = [750;750]; B0 = 100*eye(2);
phi0 = [1.3 -.7]'; iVphi = speye(2);
nu_sigc2 = 3; S_sigc2 = 1*(nu_sigc2-1);
sigtau2_ub = .01;

% storage
store_theta = zeros(nsim,6); % [phi, sigc2, sigtau2, tau0]
store_tau = zeros(nsim,T); 
store_mu = zeros(nsim,T);    % annualized trend growth

% initialize 
phi = [1.34 -.7]';
tau0 = [y(1) y(1)]'; % [tau_{0}, tau_{-1}]
sigc2 = .5;
sigtau2 = .001;

% construct a few things
S1 = sparse(2:T,1:T-1,1,T,T); % first-lag shift matrix
S2 = sparse(3:T,1:T-2,1,T,T); % second-lag shift matrix
H2 = speye(T) - 2*S1 + S2;
H2H2 = H2'*H2;
Hphi = speye(T) - phi(1)*S1 - phi(2)*S2;
Xtau0 = [(2:T+1)' -(1:T)'];
n_grid = 500;
count_phi = 0; 

for isim = 1:nsim+burnin     
    % sample tau 
    alp_tau = H2\[2*tau0(1)-tau0(2);-tau0(1);sparse(T-2,1)];    
    Ktau = H2H2/sigtau2 + Hphi'*Hphi/sigc2;
    tau_hat = Ktau\(H2H2*alp_tau/sigtau2 + Hphi'*Hphi*y/sigc2);
    tau = tau_hat + chol(Ktau,'lower')'\randn(T,1);

    % sample phi
    c = y-tau;
    Xphi = [[0;c(1:T-1)] [0;0;c(1:T-2)]];    
    Kphi = iVphi + Xphi'*Xphi/sigc2;
    phi_hat = Kphi\(iVphi*phi0 + Xphi'*c/sigc2);
    phic = phi_hat + chol(Kphi,'lower')'\randn(2,1);
    if sum(phic) < .99 && phic(2) - phic(1) < .99 ...
            && phic(2) > -.99
        phi = phic;    
        Hphi = speye(T) - phi(1)*S1 - phi(2)*S2;
        count_phi = count_phi + 1;    
    end
    
    % sample sigc2
    sigc2 = 1/gamrnd(nu_sigc2 + T/2,1/(S_sigc2 ...
        + (c-Xphi*phi)'*(c-Xphi*phi)/2));    
    
    % sample sigtau2 via Griddy-Gibbs on (0, sigtau2_ub)
    del_tau = [tau0(1); tau] ...
        - [tau0(2); tau0(1); tau(1:end-1)];
        % first differences of tau
    ddel_tau = del_tau(2:end) - del_tau(1:end-1);
        % second differences of tau
    logf_sigtau2 = @(x) -(T/2)*log(x) ...
        - (ddel_tau'*ddel_tau)./(2*x);
    sigtau2 = griddy_gibbs(logf_sigtau2, 1e-12, ...
        sigtau2_ub, n_grid);
    
    % sample tau0 
    Ktau0 = B0\speye(2) + Xtau0'*H2H2*Xtau0/sigtau2;
    tau0_hat = Ktau0\(B0\a0 + Xtau0'*H2H2*tau/sigtau2);        
    tau0 = tau0_hat + chol(Ktau0,'lower')'\randn(2,1);   

    if isim > burnin
        i = isim-burnin;
        store_tau(i,:) = tau';
        store_theta(i,:) = [phi' sigc2 sigtau2 tau0'];
        store_mu(i,:) = 4*(tau-[tau0(1);tau(1:end-1)])';
    end    
end       
tau_mean = mean(store_tau)';
theta_mean = mean(store_theta)';
theta_CI = quantile(store_theta,[.025 .975]);
mu_mean = mean(store_mu)';

% plot of graphs
tt = (1947:.25:2019.75)';
fig1 = figure; hold on;
y_gap = y - tau_mean;
yl = [min(y_gap)-1, max(y_gap)+1];
shade_nber_recessions(yl(1), yl(2));

plot(tt, y_gap, 'k', 'LineWidth', 1.5);
plot(tt, zeros(T,1), '--k', 'LineWidth', 1);

xlim([1947 2020]); ylim(yl);
box off;
xlabel('Time','FontSize',14);
ylabel('Output gap','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'Position',[100 100 900 350]);

set(gca, 'LooseInset', max(get(gca,'TightInset'), 0.02));
print(fig1, 'UC_gap', '-depsc2', '-painters');


fig2 = figure; hold on;
yl = [1 4.5];
shade_nber_recessions(yl(1), yl(2));
plot(tt, mu_mean, 'k', 'LineWidth', 1.5);
xlim([1947 2020]); ylim(yl);
box off;
xlabel('Time','FontSize',14);
ylabel('Output trend growth','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'Position',[100 100 900 350]);

set(gca, 'LooseInset', max(get(gca,'TightInset'), 0.02));
print(fig2, 'UC_trend_growth', '-depsc2', '-painters');


