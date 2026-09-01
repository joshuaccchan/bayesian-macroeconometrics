% linreg_t_gd.m
% Computes the log marginal likelihood of the AR(2) model with
% Student-t errors for US PCE inflation via the modified harmonic mean
% (Geweke-Draper) estimator. The auxiliary density f is taken as a
% multivariate normal approximation to the posterior with covariance
% Q_theta, truncated to the (1-alpha)-quantile ellipsoid of the chi^2_m
% distribution to satisfy the thin-tail condition of Geweke (1999).

linreg_t;  % estimate the t model
[nsim, m] = size(store_theta);
T = size(y,1);

% log prior
prior = @(b,s,n) lmvnpdf(b,beta0,iVbeta\speye(k)) ...
    + ligampdf(s,nu0,S0) + log(1/(nu_ub-2));

theta_hat = mean(store_theta)';
Qtheta = cov(store_theta);
alp = .05;     % significance level for truncation
chi2q = chi2inv(1-alp,m);

% Cholesky for stable logdet and quadratic forms
L = chol(Qtheta,'lower');
logdetQ = 2*sum(log(diag(L)));

% log normalizing constant for f
const_f = -0.5*m*log(2*pi) - 0.5*logdetQ - log(1-alp);

store_w = - inf(nsim,1);
for isim = 1:nsim
    theta = store_theta(isim,:)';
    s2 = (theta-theta_hat)'*(Qtheta\(theta-theta_hat));
    if s2 < chi2q
        beta = theta(1:m-2);
        sig2 = theta(m-1);
        nu = theta(m);
        e = y - X*beta;
        llike = T*(gammaln((nu+1)/2) - gammaln(nu/2)...
            - 0.5 * log(nu*pi*sig2))...
            - (nu+1)/2 * sum(log(1 + e.^2/(sig2*nu)));
        logf = const_f - 0.5*s2;  % log f(theta)
        store_w(isim) = logf ...
            - (llike + prior(beta,sig2,nu));
    end
end
maxllike = max(store_w);
log_ml = log(mean(exp(store_w-maxllike))) + maxllike;
log_ml = -log_ml;
fprintf('Log marginal likelihood: %.2f\n', log_ml);
