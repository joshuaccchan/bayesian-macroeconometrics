function h = SVAR1(ystar, h, mu, phi, sigh2)
% SVAR1.m
% Updates the log-volatility h in the stationary AR(1)
% stochastic volatility model
%     y*_t      = h_t + log(chi^2_1),
%     h_t - mu  = phi*(h_{t-1} - mu) + u_t,   u_t ~ N(0, sigh2),  |phi| < 1,
%     h_1 - mu  ~ N(0, sigh2/(1 - phi^2)),
% using the 7-component Gaussian mixture approximation of Kim, Shephard
% and Chib (1998) and the precision-based sampler of Chan and
% Jeliazkov (2009).
%
% Inputs:
% ystar:  T x 1 vector of transformed observations, y*_t = log(y_t^2 + c)
% h:      T x 1 current draw of log-volatility
% mu:     scalar; unconditional mean of h
% phi:    scalar; AR(1) persistence, |phi| < 1
% sigh2:  scalar; innovation variance
%
% Output:
% h:      T x 1 updated draw of log-volatility

T = length(h);

% 7-component normal mixture approximation for log(chi^2_1)
pj    = [0.0073 .10556 .00002 .04395 .34001 .24566 .2575];
mj    = [-10.12999 -3.97281 -8.56686 2.77786 .61942 ...
          1.79518 -1.08819] - 1.2704;
sigj2 = [5.79596 2.61369 5.17950 .16735 .64009 ...
          .34023 1.26261];
sigj  = sqrt(sigj2);

% sample mixture indicators s_t in {1,...,7}
tmprand = rand(T, 1);
q = repmat(pj, T, 1) .* normpdf(repmat(ystar, 1, 7), ...
    repmat(h, 1, 7) + repmat(mj, T, 1), repmat(sigj, T, 1));
q = q ./ repmat(sum(q, 2), 1, 7);
cdfq = cumsum(q, 2);
s = sum(tmprand > cdfq, 2) + 1;
d_s = mj(s)';
iSig_s = sparse(1:T, 1:T, 1./sigj2(s));

% sample h with precision K_h = H'*Sigma_inv*H/sigh2 + iSig_s,
% where H = I - phi*S1 and Sigma_inv = diag(1-phi^2, 1, ..., 1) encodes
% the stationary initial variance sigh2/(1-phi^2); the prior is then
% h ~ N(mu*1_T, sigh2*(H'*Sigma_inv*H)^{-1})
S1 = sparse(2:T, 1:(T-1), 1, T, T);
H  = speye(T) - phi*S1;
HiSH = H'*sparse(1:T, 1:T, [1-phi^2, ones(1, T-1)])*H;
Kh = HiSH/sigh2 + iSig_s;
h_hat = Kh\(mu/sigh2*HiSH*ones(T, 1) + iSig_s*(ystar - d_s));
CKh = chol(Kh, 'lower');
h = h_hat + CKh'\randn(T, 1);
end
