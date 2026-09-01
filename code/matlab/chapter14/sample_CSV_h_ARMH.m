function [h, accept] = sample_CSV_h_ARMH(s2, phi, sigh2, h, n, kappa)
% sample_CSV_h_ARMH.m
% Samples the common log-volatility h = (h_1,...,h_T)' in a VAR with a common
% stochastic volatility error structure (Section 14.2) using a Laplace-based
% acceptance-rejection Metropolis-Hastings (ARMH) step. The log-volatility
% follows the stationary AR(1) h_t = phi*h_{t-1} + u_t^h, u_t^h ~ N(0,sigh2),
% with h_1 ~ N(0, sigh2/(1-phi^2)). Given the per-period sums of squares s2
% (summed over the n variables), the conditional density of h is non-standard.
% A Gaussian approximation centered at the posterior mode serves as the
% proposal: a candidate is first screened by acceptance-rejection and then
% corrected by a Metropolis-Hastings step.
%
% Inputs:
%   s2:    T x 1 vector of per-period sums of squares
%   phi:   AR(1) persistence of the log-volatility
%   sigh2: innovation variance of the log-volatility
%   h:     T x 1 current draw of the log-volatility
%   n:     number of variables in the VAR
%   kappa: (optional) envelope constant for the AR screening step; larger
%          values improve mixing at the cost of more candidate draws (default 3)
%
% Outputs:
%   h:      T x 1 new draw of the log-volatility
%   accept: 1 if the MH step accepts, 0 otherwise

if nargin < 6, kappa = 3; end   % envelope constant for the AR step
T = size(h,1);
accept = 0;

% AR(1) prior precision HiSH = Hphi' * diag(.) * Hphi (zero mean)
Hphi = speye(T) - phi*sparse(2:T,1:T-1,ones(1,T-1),T,T);
HiSH = Hphi'*sparse(1:T,1:T,[(1-phi^2)/sigh2; 1/sigh2*ones(T-1,1)])*Hphi;

% step 1: locate the posterior mode by Newton-Raphson
ht = h;
max_norm = Inf; tol = 1e-4;
while max_norm > tol
    eis2 = exp(-ht).*s2;
    grad = -HiSH*ht - n/2 + 0.5*eis2;            % gradient of log posterior
    Kh = HiSH + sparse(1:T,1:T, 0.5*eis2);       % negative Hessian
    new_ht = ht + Kh\grad;
    max_norm = max(abs(new_ht - ht));
    ht = new_ht;
end
h_hat = ht;

% step 2: construct the Gaussian approximation N(h_hat, Kh^{-1})
Ch = chol(Kh,'lower');
logdetKh = 2*sum(log(diag(Ch)));

% log posterior kernel (normalizing constant omitted)
logp = @(x) -0.5*x'*HiSH*x - n/2*sum(x) - 0.5*sum(exp(-x).*s2);

% log Gaussian proposal density g(h)
logg = @(x) -0.5*T*log(2*pi) + 0.5*logdetKh - 0.5*(x-h_hat)'*Kh*(x-h_hat);

% step 3: choose c = kappa * p(h_hat)/g(h_hat)
logc = log(kappa) + logp(h_hat) - logg(h_hat);

% proposal kernel: q(h) \propto min{p(h)/(c g(h)), 1} g(h)
logq = @(x) min(logp(x) - logc - logg(x), 0) + logg(x);

% step 4: acceptance-rejection screening from g
accepted_AR = false;
while ~accepted_AR
    hc = h_hat + Ch'\randn(T,1);                 % draw from g
    if log(rand) < min(logp(hc) - logc - logg(hc), 0)
        accepted_AR = true;
    end
end

% step 5: Metropolis-Hastings correction
log_alpha = logp(hc) - logp(h) + logq(h) - logq(hc);
if log(rand) < log_alpha
    h = hc;
    accept = 1;
end
end
