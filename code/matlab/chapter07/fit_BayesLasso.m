function [beta_mean,store_beta] = ...
    fit_BayesLasso(y,X,lambda,S0,nu0,nsim,burnin)
% fit_BayesLasso.m
% Two-block Gibbs sampler for the Bayesian Lasso. The prior is the
% scale-mixture-of-normals representation of the Laplace prior:
%   (beta_j | sig2, tau_j^2) ~ N(0, sig2 * tau_j^2),
%   (tau_j^2 | lambda^2)     ~ G(1, lambda^2/2),
%   sig2 ~ IG(nu0, S0).
% In each iteration, (beta, sig2) are sampled jointly from their NIG
% full conditional, and each tau_j^2 is updated via the reciprocal
% of an inverse-Gaussian draw. Requires igaussrnd.m.
%
% Inputs:
%   y      : T-by-1 response
%   X      : T-by-k design matrix
%   lambda : Lasso shrinkage hyperparameter
%   S0,nu0 : inverse-gamma hyperparameters for sig2
%   nsim   : number of post burn-in draws
%   burnin : number of burn-in iterations
%
% Outputs:
%   beta_mean : k-by-1 posterior mean of beta
%   store_beta: nsim-by-k matrix of post burn-in beta draws
[T,k] = size(X);
XtX = X'*X; Xty = X'*y; yy = y'*y;
store_beta = zeros(nsim,k);

    % initialize
tau2 = gamrnd(1,2/lambda^2,k,1);
for isim = 1:nsim+burnin
        % sample beta and sigma^2
    Dbeta = (sparse(1:k,1:k,1./tau2) + XtX)\speye(k);
    beta_hat = Dbeta*Xty;
    S_hat = S0 + (yy -beta_hat'*(Dbeta\beta_hat))/2;
    sig2 = 1/gamrnd(nu0+T/2,1/S_hat);    
    beta = beta_hat ...
        + chol(sig2*Dbeta,'lower')*randn(k,1);
        % sample tau2
    tmp = lambda*sqrt(sig2)./abs(beta);
    tau2 = 1./igaussrnd(lambda^2*ones(k,1),tmp);
        % store draws
    if isim > burnin
        isave = isim-burnin;
        store_beta(isave,:) = beta';        
    end
end
beta_mean = mean(store_beta)';
end