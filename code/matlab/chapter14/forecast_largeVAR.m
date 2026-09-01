% forecast_largeVAR.m
% Recursive out-of-sample forecasting exercise for the large VAR
% chapter: one-step-ahead forecasts of industrial production growth,
% CPI inflation, and the unemployment rate from a 25-variable monthly
% FRED-MD VAR with p = 12 lags, re-estimated on an expanding window at
% each origin from 2000M1 to the end of the sample.
%
% Eight model-prior combinations are compared:
%   1) homoskedastic VAR, natural conjugate Minnesota prior
%   2) common stochastic volatility, natural conjugate Minnesota prior
%   3) Cholesky stochastic volatility, independent Minnesota prior
%   4) order-invariant stochastic volatility, independent Minnesota prior
%   5) factor stochastic volatility, independent Minnesota prior
%   6) order-invariant SV, Minnesota prior with estimated tightness
%      (kappa2, kappa3 sampled)
%   7) order-invariant SV, horseshoe prior
%   8) order-invariant SV, Minnesota-type horseshoe prior (estimated
%      tightness plus local horseshoe scales)
%
% The exercise is computationally intensive: with the default settings it 
% takes several hours on a multicore desktop (a parallel pool is started 
% automatically). 
% Requires pred_largeVAR_NCP.m, pred_largeVAR_CSV.m, pred_largeVAR_SV.m,
% pred_largeVAR_OISV.m, pred_largeVAR_FSV.m, and their dependencies.

clear; clc;

p = 12;    % number of lags (monthly data)
r = 4;     % number of factors in the VAR-FSV
nsim = 5000;
burnin = 1000;
run_models = 4:8;  % subset to run, e.g., run_models = [1 3]
sample_end = 2020;  % last date kept (2020.0 = 2019M12). Set to Inf to use the full sample
nworkers = 19;   % number of parallel workers; set this to roughly the
                 % number of physical cores on your machine, leaving one
                 % or two free for the system. MATLAB's default pool size
                 % can undercount the cores on hybrid CPUs (e.g., recent
                 % Intel chips), so the pool is configured explicitly below.

% load monthly FRED-MD data: the 16 variables of Carriero, Clark, Marcellino
% and Mertens (2024), augmented with nine series spanning interest rates,
% money, credit, and prices for a 25-variable panel
raw = readtable('FRED-MD.csv','VariableNamingRule','preserve');
vars = {'RPI','DPCERA3M086SBEA','INDPRO','CUMFNS','UNRATE','PAYEMS', ...
        'CES0600000007','CES0600000008','WPSFD49207','PCEPI','HOUST', ...
        'S&P 500','EXUSUKx','GS5','GS10','BAAFFM', ...  % Carriero et al. (2024)
        'FEDFUNDS','TB3MS','GS1','M2SL','BUSLOANS','CPIAUCSL', ...
        'OILPRICEx','RETAILx','PERMIT'};                % additional series
data = raw{:,vars};
% express the log-based series in percentages (100 x log-differences),
% leaving the series already in percentage points unscaled. 
in_pp = ismember(vars, {'CUMFNS','UNRATE','CES0600000007','GS5', ...
    'GS10','BAAFFM','FEDFUNDS','TB3MS','GS1'});
data(:, ~in_pp) = 100*data(:, ~in_pp);
dates = raw{:,1};
ok = all(~isnan(data),2);  % keep fully observed months
data = data(ok,:); dates = dates(ok);
pre = dates <= sample_end + 1e-6;
data = data(pre,:); dates = dates(pre);
Y0 = data(1:p,:);      % pre-sample obs (initial conditions)
Y = data(p+1:end,:);   % estimation sample
dates = dates(p+1:end);
[T,n] = size(Y);

% forecast targets: IP, CPI inflation, unemployment rate
targets = [find(strcmp(vars,'INDPRO')), find(strcmp(vars,'CPIAUCSL')), ...
    find(strcmp(vars,'UNRATE'))];
target_names = {'IP','Inflation','Unemployment'};
q = length(targets);

% prior hyperparameters
kappa1 = 100;        % intercept prior variance
kappa2 = 0.2^2;      % own-lag tightness (and overall NCP shrinkage)
kappa3 = 0.2^2/4;    % cross-lag tightness

% forecast origins: targets from 2000M1 (dates = 2000 + 1/12) to the end
% of the sample
t0 = find(dates >= 2000 + 1/12 - 1e-6, 1);
origins = t0:T;     % forecast target is Y(t,:), estimation uses Y(1:t-1,:)
nfcst = length(origins);
fcst_dates = dates(origins);
actual = Y(origins, targets);

model_names = {'Homoskedastic','Common SV','Cholesky SV', ...
    'Order-invariant SV','Factor SV', ...
    'Minnesota (estimated)','Horseshoe','Minnesota-type horseshoe'};
nmodels = length(model_names);

% res(i, m, :) = [pf(1:q), lpl(1:q), lpl_joint] for origin i, model m
res = NaN(nfcst, nmodels, 2*q+1);

% per-origin checkpointing: each completed origin is saved to disk, so an
% interrupted run can resume where it left off, and runs with different
% run_models (e.g., split across machines) are merged automatically. Each
% (origin, model) pair is seeded independently below, so the merged
% results are identical to those from a single full run
ckdir = fullfile(pwd, 'fcst_checkpoints');
if ~exist(ckdir, 'dir'), mkdir(ckdir); end
cktag = getenv('COMPUTERNAME');
if isempty(cktag), cktag = 'local'; end
cktag = sprintf('%s_m%s', cktag, sprintf('%d', run_models));

fprintf('FRED-MD: n=%d variables, %d forecast origins (%.2f-%.2f)\n', ...
    n, nfcst, fcst_dates(1), fcst_dates(end));
fprintf('models: %s\n', strjoin(model_names(run_models), '; '));

% start the parallel pool with nworkers workers. The profile cap is
% raised for this session only (the saved profile is not modified);
% on machines with fewer cores, lower nworkers above accordingly
if isempty(gcp('nocreate'))
    clu = parcluster('Processes');
    clu.NumWorkers = max(clu.NumWorkers, nworkers);
    parpool(clu, nworkers);
end

% progress counter: workers signal completed origins through a DataQueue
% and the client prints a running count with an ETA
dq = parallel.pool.DataQueue;
progress_print(0, nfcst);   % reset the counter and start the clock
afterEach(dq, @(~) progress_print(1, nfcst));

tic;
parfor ifc = 1:nfcst
    t = origins(ifc);
    Tt = t - 1;            % estimation sample: Y(1:Tt,:)
    % load any results already saved for this origin and compute only the
    % requested models that are still missing
    tmp = load_checkpoints(ckdir, ifc, nmodels, 2*q+1);
    todo = run_models(any(isnan(tmp(run_models, :)), 2));
    for m = todo
        rng(1e6 + 1000*m + t); % per-(origin, model) seed for reproducibility
        % each model is asked to forecast all n variables: the joint
        % LPL is evaluated over the full cross section, while the RMSEs
        % are computed for the three targets only
        pf = NaN(1, n); lm = NaN(1, n); lj = NaN;
        switch m
            case 1
                [pf, lj, lm] = pred_largeVAR_NCP(Y, Y0, Tt, p, ...
                    1:n, kappa1, kappa2, nsim);
            case 2
                [pf, lj, lm] = pred_largeVAR_CSV(Y, Y0, Tt, p, ...
                    1:n, kappa1, kappa2, nsim, burnin);
            case 3
                [pf, lj, lm] = pred_largeVAR_SV(Y, Y0, Tt, p, ...
                    1:n, 'minn', kappa1, kappa2, kappa3, nsim, burnin);
            case 4
                [pf, lj, lm] = pred_largeVAR_OISV(Y, Y0, Tt, p, ...
                    1:n, 'minn', kappa1, kappa2, kappa3, nsim, burnin);
            case 5
                [pf, lj, lm] = pred_largeVAR_FSV(Y, Y0, Tt, p, r, ...
                    1:n, 'minn', kappa1, kappa2, kappa3, nsim, burnin);
            case 6
                [pf, lj, lm] = pred_largeVAR_OISV(Y, Y0, Tt, p, ...
                    1:n, 'minnH', kappa1, kappa2, kappa3, nsim, burnin);
            case 7
                [pf, lj, lm] = pred_largeVAR_OISV(Y, Y0, Tt, p, ...
                    1:n, 'hs', kappa1, kappa2, kappa3, nsim, burnin);
            case 8
                [pf, lj, lm] = pred_largeVAR_OISV(Y, Y0, Tt, p, ...
                    1:n, 'mahp', kappa1, kappa2, kappa3, nsim, burnin);
        end
        tmp(m, :) = [pf(targets), lm(targets), lj];
    end
    if ~isempty(todo)
        save_checkpoint(ckdir, ifc, tmp, cktag);
    end
    res(ifc, :, :) = tmp;
    send(dq, 1);
end
fprintf('total time: %.1f minutes\n', toc/60);

% evaluation
keep = true(nfcst, 1);
RMSE = NaN(nmodels, q);
sumLPL = NaN(nmodels, 1);
for m = run_models
    err = squeeze(res(keep, m, 1:q)) - actual(keep, :);
    RMSE(m, :) = sqrt(mean(err.^2));
    sumLPL(m) = sum(res(keep, m, 2*q+1));
end

% report the two comparison tables
fprintf('\nForecast performance across SV specifications (Minnesota prior)\n');
fprintf('%-38s %8s %8s %8s %10s\n', 'Specification', target_names{:}, 'sumLPL');
for m = intersect(1:5, run_models)
    fprintf('%-38s %8.3f %8.3f %8.3f %10.1f\n', model_names{m}, ...
        RMSE(m, :), sumLPL(m));
end
fprintf('\nForecast performance across priors (order-invariant SV)\n');
fprintf('%-38s %8s %8s %8s %10s\n', 'Prior', target_names{:}, 'sumLPL');
for m = intersect([4 6 7 8], run_models)
    if m == 4, row_label = 'Minnesota'; else, row_label = model_names{m}; end
    fprintf('%-38s %8.3f %8.3f %8.3f %10.1f\n', row_label, ...
        RMSE(m, :), sumLPL(m));
end

save('forecast_largeVAR_results.mat', 'res', 'actual', 'fcst_dates', ...
    'RMSE', 'sumLPL', 'model_names', 'target_names', 'targets', ...
    'p', 'r', 'nsim', 'burnin', 'kappa1', 'kappa2', 'kappa3', 'run_models');

function tmp = load_checkpoints(ckdir, ifc, nmodels, ncols)
% merge any saved results for origin ifc, possibly written by different
% machines or by runs with different run_models subsets
tmp = NaN(nmodels, ncols);
fls = dir(fullfile(ckdir, sprintf('origin_%03d_*.mat', ifc)));
for jj = 1:numel(fls)
    S = load(fullfile(fls(jj).folder, fls(jj).name), 'tmp');
    filled = ~isnan(S.tmp);
    tmp(filled) = S.tmp(filled);
end
end

function save_checkpoint(ckdir, ifc, tmp, cktag)
% save the merged results for origin ifc; the machine- and model-specific
% file name avoids write collisions when the work is split across machines
% (the save call is wrapped in a function so it is legal inside parfor)
save(fullfile(ckdir, sprintf('origin_%03d_%s.mat', ifc, cktag)), 'tmp');
end

function progress_print(incr, total)
% running progress counter for the parfor loop: incr = 0 resets the
% count and starts the clock; incr = 1 (sent via the DataQueue each
% time a worker finishes an origin) prints the count, elapsed time,
% and an estimate of the time remaining
persistent ndone t0
if incr == 0
    ndone = 0;
    t0 = tic;
    return
end
ndone = ndone + incr;
elapsed = toc(t0)/60;
fprintf('%d of %d origins done (elapsed %.1f min, est. remaining %.1f min)\n', ...
    ndone, total, elapsed, elapsed*(total - ndone)/ndone);
end
