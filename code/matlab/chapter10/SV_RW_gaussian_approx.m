function h_hat = SV_RW_gaussian_approx(s2,h0,sigh2)
% SV_RW_gaussian_approx.m
% Posterior mean of the log-volatility vector h under a single-Gaussian
% (moment-matching) approximation to log(chi^2_1) and a random-walk prior for
% h_t. The transformed observation y*_t = log(y_t^2 + c) is treated as
% N(h_t - 1.2704, 4.9348), giving a linear Gaussian state space model whose
% posterior mean is obtained by solving a banded linear system.
%
% Inputs:
%   s2    : T-by-1 vector of squared observations y_t^2
%   h0    : initial log-volatility (state at time 0), scalar
%   sigh2 : innovation variance of the random-walk state equation, scalar
%
% Output:
%   h_hat : T-by-1 posterior mean of the log-volatility

c = 1e-4;  % avoids log(0)
mu_eps = -1.2704; % E[log chi^2_1]
var_eps = 4.9348; % Var[log chi^2_1]
T = length(s2);
S1 = sparse(2:T, 1:(T-1), 1, T, T);
H = speye(T) - S1;

ystar = log(s2 + c);
P = H'*H/sigh2;   % prior precision
b = h0*ones(T,1); % prior mean
Kh = P + speye(T)/var_eps;
h_hat = Kh\(P*b + (ystar - mu_eps)/var_eps);
end
