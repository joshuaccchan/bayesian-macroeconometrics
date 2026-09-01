% DFM.m
% Gibbs sampler for the dynamic factor model fitted to a large monthly
% macroeconomic panel (FRED-MD). The model is
%   y_t = A f_t + eps_t,                              eps_t ~ N(0, Sigma),
%   f_t = Phi_1 f_{t-1} + ... + Phi_p f_{t-p} + u_t,  u_t   ~ N(0, Omega),
% with A lower triangular (ones on the diagonal) and Sigma, Omega diagonal.
% This script estimates the one-factor, one-lag specification (r=1, p=1) as a
% business-cycle indicator. The 4-block Gibbs sampler draws the factor path f,
% the free loadings a, the variances (Sigma, Omega), and the AR coefficients
% phi (drawn from a normal truncated to the stationary region). It plots the
% posterior mean factor with NBER recession shading.
%
% Requires: shade_nber_recessions.m

clear; clc;
rng(42); % for reproducibility

nsim = 20000;
burnin = 1000;
r = 1; % number of factors

% load data
raw = readtable('FRED-MD.csv','VariableNamingRule','preserve');
dates_num = raw{:,1}; % first column contains dates as year + month/12
month_idx = round(12*dates_num); % months since year 0
year_part = floor((month_idx-1)/12);
month_part = month_idx - 12*year_part;
dates = datetime(year_part, month_part, 1);
data = raw{:,2:end}; % remaining columns are data
varnames = raw.Properties.VariableNames(2:end);

% move the 6th column (INDPRO) to the first column
perm = [6, 1:5, 7:size(data,2)];
data = data(:,perm);
varnames = varnames(perm);

% remove columns with missing values
idx = all(~isnan(data),1);
data = data(:,idx);
varnames = varnames(idx);

% standardize the data
data_mean = mean(data,1)';
data_std = std(data,0,1)';
Y = (data - data_mean') ./ data_std';

[T,n] = size(Y);
y = reshape(Y',T*n,1);

% storage
store_F = zeros(nsim,T,r);
store_A = zeros(nsim,n,r);
store_sig2 = zeros(nsim,n);
store_omega2 = zeros(nsim,r);
store_phi = zeros(nsim,r);

% prior hyperparameters
a0 = 0; Va = 1; % a_ij iid N(a0,Va)
phi0 = zeros(r,1); Vphi = ones(r,1);
nusig2 = 3; Ssig2 = (nusig2-1)*ones(n,1);
nuomega2 = 3; Somega2 = (nuomega2-1)*ones(r,1);

% initialize the Markov chain
sig2 = var(Y)';
omega2 = ones(r,1);
phi = 0.5*ones(r,1);
Phi = spdiags(phi,0,r,r);
A = [eye(r); zeros(n-r,r)];   % lower-triangular normalization

% matrices used to build H_phi
hzeros = sparse(r,(T-1)*r);
vzeros = sparse(T*r,r);
HPhi = speye(T*r) - cat(2,cat(1,hzeros,kron(speye(T-1),Phi)),vzeros);

for isim = 1:(nsim + burnin)
    % sample f
    iSig = spdiags(1./sig2,0,n,n);
    iOmega = spdiags(repmat(1./omega2,T,1),0,T*r,T*r);
    Kf = HPhi' * iOmega * HPhi + kron(speye(T), A' * iSig * A);
    f_hat = Kf \ (kron(speye(T), A' * iSig) * y);
    f = f_hat + chol(Kf,'lower')' \ randn(T*r,1);
    F = reshape(f,r,T)';

    % sample A equation by equation
    for i = 2:n
        nai = min(i-1,r);
        Xf = F(:,1:nai);
        K_ai = spdiags((1/Va)*ones(nai,1),0,nai,nai) + (Xf' * Xf) / sig2(i);
        if i <= r
            ai_hat = K_ai \ ((a0/Va)*ones(nai,1) + Xf' * (Y(:,i) - F(:,i)) / sig2(i));
        else
            ai_hat = K_ai \ ((a0/Va)*ones(nai,1) + Xf' * Y(:,i) / sig2(i));
        end
        A(i,1:nai) = ai_hat + chol(K_ai,'lower')' \ randn(nai,1);
    end

    % sample sig2
    E_y = Y - F * A';
    sig2 = 1 ./ gamrnd(nusig2 + T/2, 1 ./ (Ssig2 + sum(E_y.^2)'/2));

    % sample omega2
    E_f = [F(1,:); F(2:end,:) - F(1:end-1,:) * Phi];
    omega2 = 1 ./ gamrnd(nuomega2 + T/2, 1 ./ (Somega2 + sum(E_f.^2)'/2));

    % sample phi equation by equation (normal truncated to |phi|<1)
    Zf = [zeros(1,r); F(1:end-1,:)];
    for jj = 1:r
        Kphi_j = 1/Vphi(jj) + sum(Zf(:,jj).^2) / omega2(jj);
        phi_hat_j = (phi0(jj)/Vphi(jj) + sum(Zf(:,jj).*F(:,jj)) / omega2(jj)) / Kphi_j;
        phi_sd_j = sqrt(1/Kphi_j);
        accepted = false;
        while ~accepted
            phi_prop = phi_hat_j + phi_sd_j * randn;
            if abs(phi_prop) < 1
                phi(jj) = phi_prop;
                accepted = true;
            end
        end
    end
    Phi = spdiags(phi,0,r,r);
    HPhi = speye(T*r) - cat(2,cat(1,hzeros,kron(speye(T-1),Phi)),vzeros);

    if isim > burnin
        isave = isim - burnin;
        store_F(isave,:,:) = F;
        store_A(isave,:,:) = A;
        store_sig2(isave,:) = sig2';
        store_omega2(isave,:) = omega2';
        store_phi(isave,:) = phi';
    end

    if mod(isim,5000) == 0
        fprintf('Iteration %d of %d (%.1f%%)\n', ...
            isim, nsim + burnin, 100*isim/(nsim + burnin));
    end
end
F_mean = reshape(mean(store_F,1),T,r);

% posterior summary of the AR coefficient
phi_mean = mean(store_phi,1);
phi_q = quantile(store_phi,[0.025 0.975],1);
fprintf('Posterior mean of phi = %.3f, 95%% CI = (%.3f, %.3f)\n', ...
    phi_mean(1), phi_q(1,1), phi_q(2,1));

% decimal-year time axis for plotting
tt = year_part + (month_part - 1)/12;

% full-sample and pre-COVID plots of the posterior mean factor
idx_pre = tt <= 2019 + 11/12;          % through December 2019
tt_pre_end = tt(find(idx_pre,1,'last'));

fig = figure;
% top panel: full sample
subplot(2,1,1);
hold on;
yl_full = [min(F_mean(:,1)) - 0.2, max(F_mean(:,1)) + 0.2];
plot(tt, F_mean(:,1), 'k', 'LineWidth', 1.5);
yline(0,'k');
xlim([tt(1) tt(end)]);
ylim(yl_full);
shade_nber_recessions(yl_full(1), yl_full(2));
box off;
set(gca,'FontSize',14,'Layer','top');

% bottom panel: pre-COVID sample
subplot(2,1,2);
hold on;
yl_pre = [min(F_mean(idx_pre,1)) - 0.2, max(F_mean(idx_pre,1)) + 0.2];
plot(tt(idx_pre), F_mean(idx_pre,1), 'k', 'LineWidth', 1.5);
yline(0,'k');
xlim([tt(1) tt_pre_end]);
ylim(yl_pre);
shade_nber_recessions(yl_pre(1), yl_pre(2));
box off;
set(gca,'FontSize',14,'Layer','top');

set(gcf, 'Position', [100 100 900 600]);
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
print(fig, 'DFM_f.eps', '-depsc2', '-painters');
