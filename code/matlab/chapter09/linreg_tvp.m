% linreg_tvp.m
% Gibbs sampler for a time-varying parameter (TVP) regression, applied to a
% Phillips curve for US PCE inflation. The model is
%   y_t    = x_t'*beta_t + eps_t,     eps_t ~ N(0, sig2),
%   beta_t = beta_{t-1} + eta_t,      eta_t ~ N(0, Omega),
% with x_t = (1, gap_t, y_{t-1})' and Omega = diag(omega_1^2,...,omega_k^2), so
% each coefficient follows an independent random walk. 
% The stacked coefficient path beta is drawn in one block from its Gaussian
% full conditional, whose block-tridiagonal precision matrix uses the 
% Kronecker structure (H'H) kron Omega^{-1}; the remaining blocks 
% (sig2, omega_j^2, beta0) use standard conjugate updates. 
% The stacked design matrix Z = diag(x_1',...,x_T') is formed by SURform.m.
%
% Requires: SURform.m

clear; clc; rng(42);
nsim = 20000; burnin = 1000;

% load data
data = readmatrix('USPCE_OutputGap.csv', 'Range', 'B2:C241');
infl = data(:,1); % PCE inflation
gap  = data(:,2); % output gap

% construct y and X
y = infl(2:end);
g = gap(2:end);       
ylag = infl(1:end-1);  % y_{t-1}
T = length(y);
X = [ones(T,1), g, ylag];
k = size(X,2);

% prior hyperparameters
beta00 = zeros(k,1);
iVbeta0 = 1/100*eye(k); 
nu_sig = 3;  S_sig = 1;  % IG prior for sigma^2
nu_om = 3;               % IG prior for omega_j^2
S_om  = [0.125; 0.025; 0.025].^2*(nu_om-1);

% initialize chain
beta0 = zeros(k,1);
beta_ols = (X'*X)\(X'*y);
sig2 = mean((y - X*beta_ols).^2);
omega2 = 0.01^2 * ones(k,1); 

% precompute a few things
S1 = sparse(2:T,1:T-1,1,T,T);
H  = speye(T) - S1;
HH = H'*H; 
Z = SURform(X);
ZZ = Z'*Z;
Zy = Z'*y;

% storage 
store_beta = zeros(nsim, T*k);
    % [beta0', sig2, omega2']
store_theta = zeros(nsim, 2*k + 1); 

for isim = 1:nsim + burnin
    % sample beta    
    iOmega = sparse(1:k,1:k,1./omega2);
    P = kron(HH, iOmega); % prior precision
    Kbeta = P + ZZ/sig2;    
    CKbeta = chol(Kbeta, 'lower');            
    beta_hat = Kbeta\(P*repmat(beta0,T,1) + Zy/sig2);   
    beta = beta_hat + (CKbeta')\randn(k*T,1);

    % sample sigma^2    
    e = y - Z*beta;
    sig2 = 1/gamrnd(nu_sig + T/2, 1/(S_sig + e'*e/2));

    % sample omega_j^2
    Beta = reshape(beta,k,T)';
    SSE = sum((Beta - [beta0'; Beta(1:T-1,:)]).^2)';
    omega2 = 1./gamrnd(nu_om + T/2, 1./(S_om + 0.5*SSE));

    % sample beta0
    Kbeta0 = iVbeta0 + sparse(1:k,1:k,1./omega2);
    beta0_hat = Kbeta0\(iVbeta0*beta00 ...
        + beta(1:k)./omega2);
    Cbeta0 = chol(Kbeta0,'lower');
    beta0 = beta0_hat + (Cbeta0)'\randn(k,1);
    
    if isim > burnin
        isave = isim - burnin;
        store_beta(isave,:) = beta';
        store_theta(isave,:) = [beta0', sig2, omega2'];
    end   
end
Beta_mean = reshape(mean(store_beta, 1),k,T)';  
Beta_q = permute(reshape(quantile(store_beta, [0.05,0.95], 1)',k,T,2),[2,1,3]); 

% plot coefficients with 90% CI
tt = (1960.25:.25:2019.75)';

names = {'Intercept', 'Output gap', 'Lagged inflation'};

fig = figure('Color','w');
for j = 1:k
    subplot(k,1,j); hold on;
    lo = Beta_q(:,j,1); hi = Beta_q(:,j,2);
    fill([tt; flipud(tt)], [lo; flipud(hi)], [0.85 0.85 0.85], ...
         'EdgeColor','none', 'HandleVisibility','off');

    plot(tt, Beta_mean(:,j), 'k', 'LineWidth', 1.5);

    box off;
    set(gca,'FontSize',12,'Layer','top');
    ylabel(names{j});
    if j == k
        xlabel('Time');
    end
end
set(gcf,'Position',[100 100 600 500]);

set(gca, 'LooseInset', max(get(gca,'TightInset'), 0.02));
print(fig, 'tvp_phillips_curve', '-depsc2', '-painters');