function [hp, logintlike, info] = gpr_fit_eb(y, X, opts)
% gpr_fit_eb.m
% Empirical-Bayes hyperparameters for Gaussian process regression with an
% ARD squared exponential kernel. The model is
%   y_t = f(x_t) + eps_t,  eps_t ~ N(0, sig2),  f ~ GP(0, k),
%   k(x,x') = sigf2 * exp(-0.5 * sum_j (x_j - x'_j)^2 / lam_j^2).
% The hyperparameters (sigf2, lam_1..lam_d, sig2) are set to the maximizers
% of the log integrated likelihood, found by optimizing on the log scale (so
% positivity is automatic) from a median-heuristic start with several
% restarts. The objective is the local function gp_se_negloglike below.
%
% Inputs:
%   y    : n-by-1 response
%   X    : n-by-d input matrix
%   opts : (optional) struct with fields .nstart (restarts, default 5) and
%          .verbose (print per-restart progress, default false)
%
% Outputs:
%   hp         : struct with fields sigf2 (scalar), lam (1-by-d), sig2 (scalar)
%   logintlike : maximized log integrated likelihood
%   info       : struct with fields logp (optimizer solution) and Dc
%                (per-dimension squared-distance cell array)

if nargin < 3, opts = struct; end
if ~isfield(opts,'nstart'),  opts.nstart  = 5;     end
if ~isfield(opts,'verbose'), opts.verbose = false; end

[n,d] = size(X);

% per-dimension squared distances; median-heuristic starting length scales
Dc   = cell(1,d);
lam0 = zeros(d,1);
for j = 1:d
    Dj      = (X(:,j) - X(:,j)').^2;
    Dc{j}   = Dj;
    dv      = sqrt(Dj(triu(true(n),1)));
    lam0(j) = median(dv(dv>0));
    if ~(lam0(j) > 0), lam0(j) = 1; end
end

vy    = var(y);
logp0 = [log(vy); log(lam0); log(vy/4)];        % starting point, log scale
obj   = @(logp) gp_se_negloglike(logp, y, Dc);  % negative log int. likelihood
optfs = optimset('Display','off','MaxFunEvals',4000,'MaxIter',4000, ...
                 'TolX',1e-6,'TolFun',1e-6);

% optimize from the median-heuristic start plus perturbed restarts
best_logp = logp0;
best_nll  = Inf;
for s = 1:opts.nstart
    start = logp0;
    if s > 1, start = logp0 + 0.7*randn(d+2,1); end   % perturb on log scale
    [lp, nll] = fminsearch(obj, start, optfs);
    if nll < best_nll
        best_nll  = nll;
        best_logp = lp;
    end
    if opts.verbose
        fprintf('  restart %d: log int. lik. = %.4f\n', s, -nll);
    end
end

% unpack hyperparameters
hp.sigf2   = exp(best_logp(1));
hp.lam     = exp(best_logp(2:d+1))';        % 1-by-d length scales
hp.sig2    = exp(best_logp(d+2));
logintlike = -best_nll;
info.logp  = best_logp;
info.Dc    = Dc;
end


function nll = gp_se_negloglike(logp, y, Dc)
% Negative log integrated likelihood of the ARD squared exponential GP at the
% log-hyperparameters logp = [log(sigf2); log(lam_1..lam_d); log(sig2)].
% Integrating f out gives (y | theta) ~ N(0, Ksig) with Ksig = K + sig2*I, so
% the likelihood is evaluated through the Cholesky factor of Ksig. Returns a
% large value if Ksig stays numerically indefinite (so fminsearch backs off).
%
% Inputs:
%   logp : log-hyperparameters [log(sigf2); log(lam_1..lam_d); log(sig2)]
%   y    : n-by-1 response
%   Dc   : 1-by-d cell array of per-dimension squared-distance matrices
%          (each n-by-n)
%
% Output:
%   nll  : negative log integrated likelihood at logp

n = numel(y);
d = numel(Dc);
sigf2 = exp(logp(1));
lam   = exp(logp(2:d+1));
sig2  = exp(logp(d+2));

M = zeros(n);  % scaled squared distances
for j = 1:d
    M = M + Dc{j}/(lam(j)^2);
end
Ksig = sigf2*exp(-0.5*M) + sig2*eye(n);

[C,p] = chol(Ksig,'lower');
if p > 0        % escalate jitter if not pos. def.
    jit = 1e-8*(trace(Ksig)/n + 1);
    for k = 1:6
        [C,p] = chol(Ksig + jit*eye(n),'lower');
        if p == 0, break; end
        jit = jit*10;
    end
    if p > 0, nll = 1e10; return; end
end

a = C\y;  % a'*a = y'*Ksig^{-1}*y
logdet = 2*sum(log(diag(C))); % log|Ksig|
nll = 0.5*(a'*a) + 0.5*logdet + 0.5*n*log(2*pi);
end
