function [beta_Minn, V_Minn, s2_hat, U_hat] = ...
    Minn_indep(p, kappa1, kappa2, kappa3, Y0, Yt, rw)
% Minn_indep.m
% Constructs the hyperparameters of the independent
% Minnesota prior on the VAR coefficients,
%     beta = vec([c, A_1, ..., A_p]') ~ N(beta_Minn, diag(V_Minn)).
% Let a_{l,ij} denote the (i,j) entry of the lag-l coefficient
% matrix A_l, that is, the coefficient on lag l of variable j in
% equation i. The prior variance of each coefficient is
%     Var(intercept of equation i) = kappa1,
%     Var(a_{l,ij}) = kappa2 / l^2                if i = j (own lag),
%     Var(a_{l,ij}) = kappa3 * s_i^2 / (l^2*s_j^2) if i ~= j (cross lag),
% where s_i^2 is the residual variance from a univariate AR(p) fitted
% to variable i.
%
% Inputs:
% p:        scalar; lag order of the VAR
% kappa1:   scalar; intercept prior variance
% kappa2:   scalar; own-lag tightness
% kappa3:   scalar; cross-lag tightness
% Y0:       p0 x n matrix of pre-sample observations (p0 >= p)
% Yt:       T x n matrix of observations
% rw:       scalar; 1 = random walk prior mean (first own lag = 1),
%                   0 = zero prior mean (for growth-rate data)
%
% Outputs:
% beta_Minn: (n*k) x 1 prior mean of beta = vec(A), k = 1 + n*p
% V_Minn:    (n*k) x 1 diagonal entries of the prior covariance of beta
% s2_hat:    n x 1 vector of univariate AR(p) residual variances (s_i^2)
% U_hat:     T x n matrix of univariate AR(p) residuals

[T, n] = size(Yt);
k = 1 + n*p;

% prior mean A0: k x n
A0 = zeros(k, n);
if rw
    for j = 1:n
        A0(1 + j, j) = 1;   % coefficient on y_{j,t-1} in equation j
    end
end
beta_Minn = A0(:);
V_Minn = zeros(n*k, 1);

% estimate residual variances s_i^2 from univariate AR(p) models
s2 = zeros(n, 1);
U_hat = zeros(T, n);
tmpY = [Y0(end-p+1:end, :); Yt];
for i = 1:n
    Z_ar = ones(T, p+1);
    for l = 1:p
        Z_ar(:, l+1) = tmpY(p-l+1:end-l, i);
    end
    b_ar = Z_ar \ Yt(:, i);
    U_hat(:, i) = Yt(:, i) - Z_ar*b_ar;
    s2(i) = mean(U_hat(:, i).^2);
end
s2_hat = s2;

% prior variance for each coefficient:
%   intercept of eq. i      : kappa1
%   own lag a_{l,ii}    : kappa2 / l^2
%   cross lag a_{l,ij}  : kappa3 * s2(i) / (l^2 * s2(j))
count = 1;
for i = 1:n
    V_Minn(count) = kappa1;
    count = count + 1;
    for l = 1:p
        for j = 1:n
            if i == j
                V_Minn(count) = kappa2 / l^2;
            else
                V_Minn(count) = kappa3 * s2(i) / (l^2 * s2(j));
            end
            count = count + 1;
        end
    end
end
end
