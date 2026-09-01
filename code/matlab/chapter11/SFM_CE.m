function [logml,logml_std] = SFM_CE(store_a,store_Sig,store_Omega,Y,prior,R)
% SFM_CE.m
% Estimates the log marginal likelihood of the static factor model using the
% cross-entropy method. The importance density is a product of a Gaussian 
% for the free loadings and inverse-gamma densities for the variances, 
% with parameters matched to the posterior draws. The point estimate and 
% its numerical standard error are obtained from 20 importance batches.
%
% Requires: logintlike_SFM.m, lmvnpdf.m, ligampdf.m
%
% Inputs:
%   store_a     : posterior draws of the free loadings, nsim x na
%   store_Sig   : posterior draws of the idiosyncratic variances, nsim x n
%   store_Omega : posterior draws of the factor variances, nsim x r
%   Y           : data, T x n
%   prior       : function handle evaluating the log prior density
%   R           : number of importance samples
%
% Outputs:
%   logml     : estimated log marginal likelihood
%   logml_std : numerical standard error based on 20 importance batches

R = 20*ceil(R/20);   % make R divisible by 20 for batching
na = size(store_a,2);
n  = size(Y,2);
r  = size(store_Omega,2);

% estimate the parameters of the importance density
a_bar  = mean(store_a)';
Da_bar = cov(store_a);
Da_bar = Da_bar + 1e-10*eye(na); % small ridge for numerical stability
CDa_bar = chol(Da_bar,'lower');

nusig2_bar = zeros(n,1);
Ssig2_bar  = zeros(n,1);
for ii = 1:n
    tmp = gamfit(1./store_Sig(:,ii));
    nusig2_bar(ii) = tmp(1);
    Ssig2_bar(ii)  = 1./tmp(2);
end

nuomega2_bar = zeros(r,1);
Somega2_bar  = zeros(r,1);
for jj = 1:r
    tmp = gamfit(1./store_Omega(:,jj));
    nuomega2_bar(jj) = tmp(1);
    Somega2_bar(jj)  = 1./tmp(2);
end

% draw from the importance density
a_IS = repmat(a_bar',R,1) + (CDa_bar*randn(na,R))';

Sig_IS = zeros(R,n);
for ii = 1:n
    Sig_IS(:,ii) = 1./gamrnd(nusig2_bar(ii),1./Ssig2_bar(ii),R,1);
end

Omega_IS = zeros(R,r);
for jj = 1:r
    Omega_IS(:,jj) = 1./gamrnd(nuomega2_bar(jj),1./Somega2_bar(jj),R,1);
end

% log importance density
g_IS = @(ax,s,o) lmvnpdf(ax,a_bar,Da_bar) ...
    + sum(ligampdf(s,nusig2_bar,Ssig2_bar)) ...
    + sum(ligampdf(o,nuomega2_bar,Somega2_bar));

% Log importance weights
store_w = zeros(R,1);
for isim = 1:R
    a     = a_IS(isim,:)';
    Sig   = Sig_IS(isim,:)';
    Omega = Omega_IS(isim,:)';

    llike = logintlike_SFM(Y,a,Sig,Omega);
    store_w(isim) = llike + prior(a,Sig,Omega) - g_IS(a,Sig,Omega);
end

% Batch estimate of log marginal likelihood
shortw = reshape(store_w,R/20,20);      % 20 batches
maxw   = max(shortw,[],1);              % batch-specific maxima
bigml  = log(mean(exp(shortw - repmat(maxw,R/20,1)),1)) + maxw;

logml     = mean(bigml);
logml_std = std(bigml)/sqrt(20);

end
