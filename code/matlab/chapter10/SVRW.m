function h = SVRW(ystar, h, h0, sigh2)
% SVRW.m
% One MCMC update of the log-volatility vector h in the random-walk stochastic
% volatility model
%   y*_t = h_t + log(chi^2_1),
%   h_t  = h_{t-1} + u_t,   u_t ~ N(0, sigh2),   h_1 = h0 + u_1,
% using the 7-component Gaussian mixture approximation of Kim, Shephard and
% Chib (1998) and the precision-based sampler of Chan and Jeliazkov (2009). 
% The mixture indicators are drawn first, then h is drawn from its Gaussian full
% conditional with a banded precision matrix. 
%
% Inputs:
%   ystar : T-by-1 vector of transformed observations, y*_t = log(y_t^2 + c)
%   h     : T-by-1 current draw of the log-volatility
%   h0    : initial log-volatility (state at time 0), scalar
%   sigh2 : innovation variance of the random-walk state equation, scalar
%
% Output:
%   h     : T-by-1 updated draw of the log-volatility

T = length(h);

% 7-component normal mixture approximation for log(chi^2_1)
pj    = [0.0073 .10556 .00002 .04395 .34001 .24566 .2575];
mj    = [-10.12999 -3.97281 -8.56686 2.77786 .61942 ...
          1.79518 -1.08819] - 1.2704;
sigj2 = [5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261];
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

% sample h
S1 = sparse(2:T, 1:(T-1), 1, T, T);
H  = speye(T) - S1;
HH = H'*H;
Kh = HH/sigh2 + iSig_s;
h_hat = Kh\(h0/sigh2*HH*ones(T, 1) + iSig_s*(ystar - d_s));
CKh = chol(Kh, 'lower');
h = h_hat + CKh'\randn(T, 1);
end
