function [A0, VA, nu0, S0] = Minn_NCP(Y, Y0, p, kappa1, kappa2, rw)
% Minn_NCP.m
% Constructs the natural conjugate (normal-inverse-Wishart) prior
% hyperparameters (A, Sigma) ~ NIW(A0, VA, nu0, S0) using Minnesota-style
% elicitation, with residual variances from univariate AR(p) fits.
%
% Inputs:
%   Y      : T x n matrix of observations
%   Y0     : p0 x n matrix of pre-sample observations (p0 >= p)
%   p      : lag order
%   kappa1 : prior variance on the intercepts
%   kappa2 : overall shrinkage on the lag coefficients
%   rw     : 1 = random-walk prior mean (first own lag = 1),
%            0 = zero prior mean (for growth-rate data)
%
% Outputs:
%   A0  : k x n prior mean of the coefficient matrix A, k = 1+n*p
%   VA  : k x k diagonal prior scale matrix for vec(A)
%   nu0 : prior degrees of freedom for the inverse-Wishart
%   S0  : n x n prior scale matrix for the inverse-Wishart

[~, n] = size(Y);
k = 1 + n*p;

% estimate residual variances from univariate AR(p) models
s2 = zeros(n, 1);
for i = 1:n
    tmpY = [Y0(end-p+1:end, i); Y(:, i)];
    Z_ar = zeros(size(Y, 1), p);
    for j = 1:p
        Z_ar(:, j) = tmpY(p-j+1:end-j);
    end
    Z_ar = [ones(size(Y, 1), 1), Z_ar];
    b_ar = Z_ar \ Y(:, i);
    e_ar = Y(:, i) - Z_ar * b_ar;
    s2(i) = mean(e_ar.^2);
end

% prior mean A0: k x n
A0 = zeros(k, n);
if rw
    for j = 1:n
        A0(1 + j, j) = 1;   % coefficient on y_{j,t-1} in equation j
    end
end

% prior scale matrix VA: k x k diagonal
% intercept: kappa1
% lag l, variable r: kappa2 / (l^2 * s_r^2)
va = zeros(k, 1);
va(1) = kappa1;              % intercept
for l = 1:p
    for r = 1:n
        idx = 1 + (l-1)*n + r;
        va(idx) = kappa2 / (l^2 * s2(r));
    end
end
VA = diag(va);

% inverse-Wishart hyperparameters
nu0 = n + 2;
S0 = diag(s2);
end
