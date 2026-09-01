function lml = ml_VAR_ACP(Y, Y0, p, kappa, s2)
% ml_VAR_ACP.m
% Evaluates the log marginal likelihood of a VAR under the asymmetric
% conjugate prior of Chan (2022), computed equation by equation in the
% recursive structural form (with a zero coefficient prior mean).
%
% Inputs:
%   Y     : T x n matrix of observations
%   Y0    : p0 x n matrix of pre-sample observations (p0 >= p)
%   p     : lag order
%   kappa : [kappa1 kappa2 kappa3] intercept, own-lag and other-lag shrinkage
%   s2    : n x 1 residual variances from univariate AR(p) models
%
% Output:
%   lml : log marginal likelihood

[T, n] = size(Y);

% build the lag regressor matrix W (intercept and p lags)
tmpY = [Y0(end-p+1:end,:); Y];
Z = zeros(T, n*p);
for l = 1:p
    Z(:, (l-1)*n+1:l*n) = tmpY(p-l+1:end-l, :);
end
W = [ones(T,1) Z];

% accumulate the equation-by-equation contributions
lml = -T*n/2*log(2*pi);
for i = 1:n
    yi = Y(:,i);
    % regressors: contemporaneous variables and lags
    Xi = [-Y(:,1:i-1), W];

    % prior scale V_i = diag(V_alpha, V_beta), with the
    % Minnesota own- and other-lag shrinkage in V_beta
    vb = zeros(1+n*p, 1);  vb(1) = kappa(1);
    for l = 1:p
        for r = 1:n
            idx = 1 + (l-1)*n + r;
            if r == i
                vb(idx) = kappa(2)/(l^2*s2(i));   % own lag
            else
                vb(idx) = kappa(3)/(l^2*s2(r));   % other lag
            end
        end
    end
    Vi = [1./s2(1:i-1); vb];
    nu_i = 1 + i/2;  S_i = s2(i)/2;

    % posterior quantities (the prior mean is zero)
    K  = diag(1./Vi) + Xi'*Xi;
    CK = chol(K, 'lower');
    th = CK'\(CK\(Xi'*yi));
    S_hat = S_i + (yi'*yi - th'*K*th)/2;

    % add the contribution of equation i
    lml = lml - 0.5*(sum(log(Vi)) + 2*sum(log(diag(CK)))) ...
        + gammaln(nu_i+T/2) + nu_i*log(S_i) - gammaln(nu_i) ...
        - (nu_i+T/2)*log(S_hat);
end
end
