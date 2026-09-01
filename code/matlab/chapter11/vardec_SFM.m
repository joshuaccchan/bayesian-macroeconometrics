function [vd_mean, sys_mean, idio_mean] = vardec_SFM(store_a,store_sig2,store_omega2)
% vardec_SFM.m
% Computes posterior mean variance decompositions for the static factor model.
% For each draw it reconstructs the loading matrix A, forms the factor-specific
% variance contributions a_{ij}^2 * omega_j^2 and the idiosyncratic variance
% sigma_i^2, and converts them into shares of each series' total variance.
%
% Inputs:
%   store_a      : posterior draws of the free loadings, nsim x na
%   store_sig2   : posterior draws of diag(Sigma), nsim x n
%   store_omega2 : posterior draws of diag(Omega), nsim x r
%
% Outputs:
%   vd_mean   : posterior mean variance shares, n x (r+1); columns 1..r are the
%               factor shares and column r+1 is the idiosyncratic share
%   sys_mean  : posterior mean share due to all common factors, n x 1
%   idio_mean : posterior mean idiosyncratic share, n x 1

nsim = size(store_a,1);
n = size(store_sig2,2);
r = size(store_omega2,2);
vd_store = zeros(nsim,n,r+1);

for isim = 1:nsim
    a = store_a(isim,:)';
    sig2 = store_sig2(isim,:)';
    omega2 = store_omega2(isim,:)';

    % reconstruct lower-triangular loading matrix with ones on diagonal
    A = [eye(r); zeros(n-r,r)];
    count_a = 0;
    for ii = 2:n
        nai = min(ii-1,r);
        A(ii,1:nai) = a(count_a+1:count_a+nai);
        count_a = count_a + nai;
    end

    % factor-specific variance contributions: n x r
    fac_var = A.^2 .* repmat(omega2',n,1);

    % idiosyncratic variance contribution: n x 1
    idio_var = sig2;

    % total variance of each series: n x 1
    total_var = sum(fac_var,2) + idio_var;

    % variance shares
    vd_store(isim,:,1:r) = fac_var ./ total_var;
    vd_store(isim,:,r+1) = idio_var ./ total_var;
end

vd_mean = squeeze(mean(vd_store,1));
sys_mean = sum(vd_mean(:,1:r),2);
idio_mean = vd_mean(:,r+1);
end
