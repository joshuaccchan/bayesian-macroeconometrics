% linreg_ma1_sddr.m
% Computes the Bayes factor for the ARMA(2,1) model against the AR(2)
% model with independent errors via the Savage-Dickey density ratio.
% The prior for psi is uniform on (-1, 1), so the prior ordinate at
% psi = 0 is 0.5. The posterior ordinate at psi = 0 is estimated by
% averaging the conditional posterior densities p(psi | y, beta, sig2)
% over the posterior draws of (beta, sig2) produced by linreg_ma1.m.

linreg_ma1;   % estimate the ARMA(2,1) model
n_grid = 400; % number of grid points
psi_grid = linspace(-.99,.99,n_grid)'; % grid for psi
psi_grid = sort([psi_grid; 0]); % insert 0 explicitly
n_grid = size(psi_grid,1);
idx_0 = find(psi_grid == 0); % index for psi = 0
lp_psi = zeros(n_grid,1);    % log posterior density
store_p_psi = zeros(n_grid,1);

for isim = 1:nsim
    beta = store_theta(isim,1:3)';
    sig2 = store_theta(isim,4);

    for igrid = 1:n_grid
        psi = psi_grid(igrid);
        lp_psi(igrid) = loglike_MA1(psi,...
            y-X*beta, sig2);
    end
    % normalize to obtain conditional posterior of psi
    p_psi = exp(lp_psi - max(lp_psi));
    p_psi = p_psi / trapz(psi_grid, p_psi);
    store_p_psi = store_p_psi + p_psi;
end
p_psi_hat = store_p_psi / nsim; % posterior of psi
BF_UR = 0.5 / p_psi_hat(idx_0); % BF in favor of MA(1)
fprintf('Bayes factor in favor of MA(1): %.3f\n',...
    BF_UR);

% plot prior and posterior densities
prior_psi = 0.5 * ones(n_grid,1);
figure('Position',[200 200 500 350]); hold on;
plot(psi_grid, p_psi_hat, 'k-', 'LineWidth', 2);
plot(psi_grid, prior_psi, 'k--', 'LineWidth', 1.5);
box off; set(gca,'FontSize',14);
xlabel('$\psi$','Interpreter','latex');
ylabel('Density','Interpreter','latex');
legend({'Posterior','Prior'}, 'Location','best');
