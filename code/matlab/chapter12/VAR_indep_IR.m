% VAR_indep_IR.m
% Structural impulse response analysis of the oil market using a VAR(p) with
% an independent normal and inverse-Wishart prior. A two-block Gibbs 
% sampler draws the VAR coefficients and the error covariance; at each 
% post-burn-in draw the impulse responses to the three structural shocks
% (recursively identified) are computed with construct_IR.m, and the 
% posterior means and pointwise 90% credible bands are plotted.
%
% Requires: construct_IR.m, plotCI.m

clear; clc;
rng(42); % for reproducibility

p = 24;
nsim = 10000;
burnin = 1000;
n_hz = 19;     % impulse response horizon: 0 to 18 months

% load data
data = readmatrix('oil_SVAR_data.csv', 'NumHeaderLines', 1);
Y_all = data(:, 2:4);   % full sample: 1973M2-2019M12
Y0 = Y_all(1:p, :);     % initial conditions
Y = Y_all(p+1:end, :);  % estimation sample
[T, n] = size(Y);
k = 1 + n*p;     % number of coefficients per equation

% prior hyperparameters
% prior mean: variable 1 (growth rate) -> 0;
% variables 2,3 (levels) -> random walk
beta0 = zeros(n*k, 1);
for j = 2:n
    beta0((j-1)*k + 1 + j) = 1; % first own lag = 1 for level variables
end
iVbeta = 1/100*speye(n*k);
nu0 = n + 2;  S0 = eye(n);

% construct the T x k regressor matrix Z
tmpY = [Y0(end-p+1:end, :); Y];
Z = zeros(T, n*p);
for i = 1:p
    Z(:, (i-1)*n+1:i*n) = tmpY(p-i+1:end-i, :);
end
Z = [ones(T, 1), Z];
ZZ = Z'*Z; ZY = Z'*Y;

% initialize the Gibbs sampler at OLS
A = Z \ Y;
beta = A(:);
E = Y - Z*A;
Sig = E'*E/T;
iSig = Sig \ speye(n);

% storage for impulse responses: nsim x n_hz x n variables x n shocks
store_yIR = zeros(nsim, n_hz, n, n);

for isim = 1:nsim + burnin
    % sample beta
    Kbeta = iVbeta + kron(iSig, ZZ);
    Cbeta = chol(Kbeta, 'lower');
    beta_hat = Cbeta' \ (Cbeta \ (iVbeta*beta0 + reshape(ZY*iSig, n*k, 1)));
    beta = beta_hat + Cbeta' \ randn(n*k, 1);

    % sample Sigma
    E = Y - Z*reshape(beta, k, n);
    Sig = iwishrnd(S0 + E'*E, nu0 + T);
    iSig = Sig \ speye(n);

    % compute impulse responses
    if isim > burnin
        isave = isim - burnin;
        for jj = 1:n
            % normalize: each shock raises the oil price
            % shock 1 (supply): negative supply shock -> raise price
            % shocks 2,3 (demand): positive demand shock -> raise price
            shock = zeros(n, 1);
            if jj == 1
                shock(jj) = -1;   % negative oil supply shock
            else
                shock(jj) = 1;    % positive demand shock
            end
            yIR = construct_IR(beta, Sig, n_hz, shock);
            store_yIR(isave, :, :, jj) = yIR;
        end
    end
end

% cumulate oil production responses (variable 1 is a growth rate)
store_yIR(:,:,1,:) = cumsum(store_yIR(:,:,1,:), 2);

% posterior mean and 90% credible intervals
yIR_mean = squeeze(mean(store_yIR));
yIR_lo = squeeze(quantile(store_yIR, .05));
yIR_hi = squeeze(quantile(store_yIR, .95));

% plot: 3 x 3 grid (rows = variables, columns = shocks)
varnames = {'Oil production', 'Real activity', 'Real price of oil'};
shocknames = {'Oil supply shock', 'Aggregate demand shock', ...
              'Oil-specific demand shock'};
hz = (0:n_hz-1)';

figure;
for ii = 1:n       % response variable (row)
    for jj = 1:n   % shock (column)
        subplot(n, n, (ii-1)*n + jj);
        hold on
            plotCI(hz, yIR_lo(:,ii,jj), yIR_hi(:,ii,jj));
            plot(hz, yIR_mean(:,ii,jj), 'k', 'LineWidth', 1.5);
            yline(0, 'k-', 'LineWidth', 0.5);
        hold off
        box off;
        xlim([-0.5 n_hz-1]);
        yl = ylim;
        ylim([min(yl(1), -0.5) max(yl(2), 0.5)]);
        if ii == 1; title(shocknames{jj}); end
        if jj == 1; ylabel(varnames{ii}); end
        if ii == n; xlabel('Months'); end
    end
end
set(gcf, 'Position', [100 100 800 400]);

print(gcf, 'oil_SVAR_IR', '-depsc');
