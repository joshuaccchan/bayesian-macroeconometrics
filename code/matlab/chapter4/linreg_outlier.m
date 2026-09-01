% linreg_outlier.m
% Gibbs sampler for an AR(2) model of US PCE inflation with a
% discrete outlier component:
%   y_t = beta_1 + beta_2 y_{t-1} + beta_3 y_{t-2} + o_t * eps_t,
%   (eps_t | sig2) ~ N(0, sig2),
% where o_t in {1, 5, 10}, P(o_t=1) = 1-p_o, P(o_t=5) = P(o_t=10) =
% p_o/2. The four blocks are: (beta, sig2) from conjugate updates
% conditional on the latent scales, the discrete indicators {o_t}
% from their posterior over {1, 5, 10}, and the outlier probability
% p_o from its beta full conditional.

clear; clc; rng(42);
nsim = 50000; burnin = 1000;

% load data
data = readmatrix('USPCE.csv', 'Range', 'B2:B261');
y0 = data(1:4);
y  = data(5:end);
T  = length(y);
xlag1 = [y0(4); y(1:end-1)];
xlag2 = [y0(3:4); y(1:end-2)];
X = [ones(T,1), xlag1, xlag2];
k = size(X,2);

% prior hyperparameters
beta0 = zeros(k,1); iVbeta = speye(k)/100;
nu0 = 3; S0 = 2;
p0a = 10/4; p0b = (1-1/16)*40;

% initialize the chain
beta = (X'*X)\(X'*y);
sig2 = sum((y-X*beta).^2)/T;
po = 1/16;  
o = ones(T,1);

store_o = zeros(nsim,T);
store_theta = zeros(nsim,5);  % [beta', sig2, po]

o_grid = [1, 5, 10]';   % possible values for o_t
for isim = 1:nsim + burnin
        % sample beta
    iO = sparse(1:T,1:T,1./o.^2);    
    Dbeta = (iVbeta + X'*iO*X/sig2)\speye(k); 
    beta_hat = Dbeta*(iVbeta*beta0 + X'*iO*y/sig2);
    beta = beta_hat + chol(Dbeta,'lower')*randn(k,1);
  
        % sample sig2
    e = (y - X*beta)./o;
    sig2 = 1/gamrnd(nu0+T/2,1/(S0 + e'*e/2)); 

        % sample o
    o_lpri = log([1-po; po/2; po/2]);   % log prior density
    u = (y - X*beta)/sqrt(sig2);
    for tt=1:T
        lliket = -log(o_grid) -.5*u(tt)^2./o_grid.^2;
        o_post = exp(lliket + o_lpri - max(lliket));
        o_post = o_post/sum(o_post);
        idx = find(rand<cumsum(o_post),1);
        o(tt) = o_grid(idx);
    end

        % sample po
    tmp = sum(o>1);
    po = betarnd(p0a + tmp, p0b + T-tmp);
    
    if (mod(isim, 5000) == 0)
        disp([num2str(isim) ' loops... ']);
    end
    
        % store the parameters
    if isim > burnin
        isave = isim - burnin;
        store_theta(isave,:) = [beta', sig2, po];
        store_o(isave,:) = o';
    end
end
theta_mean = mean(store_theta);
theta_CI = quantile(store_theta,[.025 .975]);
o_mean = mean(store_o)';

% Outlier probability: posterior mean and pointwise 95% CI
Iout = (store_o > 1);                    % nsim-by-T logical
p_mean = mean(Iout, 1)';                 % T-by-1 posterior mean

p_lo95 = quantile(Iout, 0.025, 1)';      % T-by-1 (pointwise lower)
p_hi95 = quantile(Iout, 0.975, 1)';      % T-by-1 (pointwise upper)

tgrid = (1:T)';

figure; hold on;

% 95% pointwise credible band
h_band = shaded_band(tgrid, p_lo95, p_hi95, 0.85);
set(h_band, 'HandleVisibility', 'off');

% posterior mean
plot(tgrid, p_mean, 'k-', 'LineWidth', 1.5, ...
    'DisplayName', 'Posterior mean');

hold off; box off;

xlabel('$t$', 'Interpreter', 'latex');
ylabel('$\mathbb{P}(o_t>1\mid \mathbf{y})$', 'Interpreter', 'latex');

set(gca, 'FontSize', 14);
set(gcf, 'Color', 'w');
legend('Location','best');
ylim([0 1]);

figure;

subplot(2,1,1);
plot(tgrid, y, 'k-', 'LineWidth', 1);
box off;
ylabel('Inflation');
set(gca, 'FontSize', 14);

subplot(2,1,2);
hold on;
shaded_band(tgrid, p_lo95, p_hi95, 0.85);
plot(tgrid, p_mean, 'k-', 'LineWidth', 1.5);
hold off; box off;

xlabel('$t$', 'Interpreter', 'latex');
ylabel('$\mathbb{P}(o_t>1\mid \mathbf{y})$', 'Interpreter', 'latex');
set(gca, 'FontSize', 14);
ylim([0 1]);

set(gcf, 'Color', 'w');
set(gcf, 'Position', [100 100 800 400]);

