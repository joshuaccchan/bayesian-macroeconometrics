% infmix_logchi2.m
% Requires: tpdfLS.m

clear; clc;

nsim   = 10000;
burnin = 1000;

% generate data
rng(1);
T = 2000;
df = 1;
y = log(chi2rnd(df,T,1));

% true log-chi^2_1 density
ftrue = @(x) 1/sqrt(2*pi) * exp(0.5*x - 0.5*exp(x));

% prior hyperparameters
nu0 = 2;  S0  = 1;
mu0 = 0;  Vmu = 100;
a_alpha = 1; b_alpha = 1; % prior on alpha

% grid
ngrid = 500;
xgrid = linspace(-10,5,ngrid)';
store_mixden = zeros(nsim, ngrid);
store_M = zeros(nsim,1);

% initialize using K-means
Minit = 10;
[s, ~] = kmeans(y, Minit);
alpha = gamrnd(a_alpha,1/b_alpha);

% maintain clusters dynamically using sufficient statistics
% for cluster m: T(m), sumy(m), sumy2(m)
M = max(s);
Tm = zeros(M,1);
sumy = zeros(M,1);
sumy2 = zeros(M,1);

for m=1:M
    idx = (s==m);
    Tm(m) = sum(idx);
    ym = y(idx);
    sumy(m)= sum(ym);
    sumy2(m)= ym'*ym;
end

% posterior predictive density
pred_t_density = @(z, Tm, sy, sy2)...
    local_pred_t(z, Tm, sy, sy2, mu0, Vmu, nu0, S0);

for isim = 1:(nsim + burnin)
    % sample alpha
    M = length(Tm); % current # of clusters
    eta = betarnd(alpha+1, T);
    pi_eta = (a_alpha + M - 1)/(T*(b_alpha ...
        - log(eta))+ a_alpha + M - 1);
    if rand < pi_eta
        alpha = gamrnd(a_alpha + M, ...
            1/(b_alpha - log(eta)));
    else
        alpha = gamrnd(a_alpha + M - 1, ...
            1/(b_alpha - log(eta)));
    end    

    % sample s_t
    for t = 1:T
        M = length(Tm);
        mt = s(t);
        % remove observation t from its current cluster
        Tm(mt) = Tm(mt) - 1;
        sumy(mt) = sumy(mt) - y(t);
        sumy2(mt) = sumy2(mt) - y(t)^2;

        % if cluster becomes empty, remove it
        if Tm(mt) == 0
            % move last cluster to mt (if mt not last)
            if mt ~= M 
                Tm(mt)= Tm(M);
                sumy(mt) = sumy(M);
                sumy2(mt) = sumy2(M);

                % relabel all obs. assigned to M to mt
                s(s==M) = mt;
            end
            % shrink arrays
            Tm(M) = []; sumy(M)=[]; sumy2(M)=[];
            M = M - 1;
        end

        % compute assignment probabilities
        logp = zeros(M+1,1);
        for m = 1:M
            fm = pred_t_density(y(t), Tm(m), ...
                sumy(m), sumy2(m));
            % (eps, realmin) added to avoid log(0)
            logp(m) = log(Tm(m) + eps)...
                + log(fm + realmin);
        end
        
        % prior predictive (new cluster) density f0(y_t)
        f0 = pred_t_density(y(t), 0, 0, 0);
        logp(M+1) = log(alpha + eps) + log(f0 + realmin);

        % normalize safely
        logp = logp - max(logp);
        p = exp(logp);
        p = p / sum(p);

        % sample new assignment        
        newm = find(rand <= cumsum(p), 1, 'first');

        if newm <= M
            % assign to existing cluster
            s(t) = newm;
            Tm(newm) = Tm(newm) + 1;
            sumy(newm) = sumy(newm) + y(t);
            sumy2(newm) = sumy2(newm) + y(t)^2;
        else
            % create new cluster
            M = M + 1;
            s(t) = M;
            Tm(M,1) = 1;
            sumy(M,1) = y(t);
            sumy2(M,1) = y(t)^2;
        end
    end

    if isim > burnin
        isave = isim - burnin;
        % posterior predictive
        f0g = pred_t_density(xgrid, 0, 0, 0);
        mixden = (alpha/(alpha+T)) * f0g;
        for m = 1:M
            fm_g = pred_t_density(xgrid, Tm(m),...
                sumy(m), sumy2(m));
            mixden = mixden + (Tm(m)/(alpha+T)) * fm_g;
        end
        store_mixden(isave,:) = mixden';
        store_M(isave) = M;
    end
end

% posterior summaries
M_mean = mean(store_M);
mixden_mean = mean(store_mixden,1)';
mixden_low  = prctile(store_mixden, 2.5, 1)';
mixden_high = prctile(store_mixden, 97.5, 1)';

% plot
fig = figure; hold on;

fill([xgrid; flipud(xgrid)], [mixden_low; flipud(mixden_high)], ...
    [0.85 0.85 0.85], 'EdgeColor', 'none', 'HandleVisibility', 'off');

hMean = plot(xgrid, mixden_mean, 'k-', 'LineWidth', 2, ...
    'DisplayName', 'DPM');

hTrue = plot(xgrid, ftrue(xgrid), 'k--', 'LineWidth', 2, ...
    'DisplayName', 'true density');

hold off; box off;
xlim([-10 5]);

xlabel('$y$', 'Interpreter', 'latex');
ylabel('Density');

set(gca, 'FontSize', 14);
set(gcf, 'Color', 'w');

legend([hMean, hTrue], 'Location', 'best', 'FontSize', 14);

% Tight layout
set(gca, 'LooseInset', max(get(gca,'TightInset'), 0.02));

% Save as EPS with embedded fonts
print(fig, 'infmixture_logchi2', '-depsc2', '-painters');

function f = local_pred_t(z, Tm, sumy, sumy2, mu0, Vmu, nu0, S0)
% Posterior predictive density of a normal-inverse-gamma component evaluated
% at z. Integrating out (mu, sig2) under the conjugate base measure G_0 yields
% a Student-t; setting Tm = 0 returns the prior predictive f_0 under G_0.
%
% Inputs:
%   z    : scalar or vector of evaluation points (e.g. the grid)
%   Tm   : number of observations currently in the cluster
%   sumy : sum of the cluster observations, sum_t y_t
%   sumy2: sum of squared cluster observations, sum_t y_t^2
%   mu0  : prior mean of mu
%   Vmu  : prior variance scale, mu | sig2 ~ N(mu0, sig2*Vmu)
%   nu0  : prior shape for sig2, sig2 ~ IG(nu0, S0)
%   S0   : prior scale for sig2
%
% Output:
%   f    : predictive density evaluated at z (same size as z)
    Kmu = 1/Vmu + Tm;
    nu_hat = nu0 + Tm/2;
    if Tm == 0
        mu_hat = mu0;
        S_hat  = S0;
    else
        mu_hat = (mu0/Vmu + sumy) / Kmu;
        S_hat  = S0 + 0.5*(sumy2 + mu0^2/Vmu - Kmu*mu_hat^2);
    end    
    s2 = (S_hat / nu_hat) * (1 + 1/Kmu);
    nu = 2*nu_hat;

    f = tpdfLS(z, mu_hat, s2, nu);    
end

