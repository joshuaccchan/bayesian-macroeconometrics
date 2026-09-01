% finmix_logchi2.m
% Gibbs sampler that fits a four-component normal mixture
%   y_t ~ sum_{m=1}^M w_m * N(mu_m, sig2_m)
% to simulated data from the log chi^2_1 distribution. Component
% indicators are sampled via the inverse-transform method, mixture
% weights from their Dirichlet full conditional (dirirnd.m), and
% (mu_m, sig2_m) from their normal-inverse-gamma full conditional.
% The chain is initialized using K-means clustering.

clear; clc; rng(42);

nsim = 20000; burnin = 1000;
M = 4; % # of components

% generate data
T = 2000;
df = 1;
y = log(chi2rnd(df,T,1));

% true log-chi^2_1 density
ftrue = @(x) 1/sqrt(2*pi) * exp(0.5*x - 0.5*exp(x));

% prior hyperparameters
nu0 = 2; S0 = 1;
mu0 = 0; Vmu = 100;
a0 = 2*ones(M,1);

% grid
ngrid = 500;
xgrid = linspace(-10,5,ngrid)';

store_mixden = zeros(nsim,ngrid);
like = zeros(T,M);

% initialize using k-means
[s, mu] = kmeans(y, M);
sig2 = zeros(M,1);
alp  = zeros(M,1);

for m = 1:M
    idx = (s == m);
    sig2(m) = var(y(idx));
    alp(m)  = sum(idx)/T; 
end

for isim = 1:(nsim + burnin)
    % sample (mu_m, sig2_m) for each component
    for m = 1:M
        idx = (s == m);
        Tm  = sum(idx);
        ym  = y(idx);

        Kmum   = 1/Vmu + Tm;
        mum_hat = (mu0/Vmu + sum(ym))/Kmum;
        Sm_hat  = S0 + 0.5*(ym'*ym + mu0^2/Vmu...
            - mum_hat^2*Kmum);

        sig2(m) = 1/gamrnd(nu0 + Tm/2, 1/Sm_hat);
        mu(m)   = mum_hat + sqrt(sig2(m)/Kmum)*randn;
    end

    % sample s
    for m = 1:M
        like(:,m) = normpdf(y, mu(m), sqrt(sig2(m)));
    end
    joint_den = like .* repmat(alp', T, 1);
    prob = joint_den ./ repmat(sum(joint_den,2), 1, M);
    u = rand(T,1);
    cumprob = cumsum(prob, 2);
    s = 1 + sum(u > cumprob, 2);

    % sample alp
    ns = zeros(M,1);
    for m = 1:M
        ns(m) = sum(s == m);
    end
    alp = dirirnd(ns + a0, 1)';

    % store mixture density
    if isim > burnin
        isave = isim - burnin;
        mixden = normpdf(xgrid, mu', sqrt(sig2')) * alp; 
        store_mixden(isave,:) = mixden';
    end
end
mixden_mean = mean(store_mixden,1)';
mixden_low  = prctile(store_mixden, 2.5, 1)';
mixden_high = prctile(store_mixden, 97.5, 1)';

figure; hold on;

% 95% pointwise posterior credible band
h_band = shaded_band(xgrid, mixden_low, mixden_high, 0.85);
set(h_band, 'HandleVisibility', 'off');

% posterior mean (solid black)
hMean = plot(xgrid, mixden_mean, 'k-', 'LineWidth', 2, ...
    'DisplayName', 'finite mixture');

% true density (dashed black)
hTrue = plot(xgrid, ftrue(xgrid), 'k--', 'LineWidth', 2, ...
    'DisplayName', 'true density');

hold off; box off;
xlim([-10 5]);

xlabel('$y$', 'Interpreter', 'latex');
ylabel('Density');

set(gca, 'FontSize', 14);
set(gcf, 'Color', 'w');

legend([hMean, hTrue], 'Location', 'best', 'FontSize', 14);
