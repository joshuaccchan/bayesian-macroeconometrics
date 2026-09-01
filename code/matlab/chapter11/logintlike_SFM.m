function loglike = logintlike_SFM(Y,a,Sig,Omega)
% logintlike_SFM.m
% Evaluates the log integrated likelihood of the static factor model, i.e., the
% Gaussian density of the data with the latent factors integrated out:
%   y_t ~ N(0, A*Omega*A' + Sigma).
% The n x n inverse is obtained via the Sherman-Morrison-Woodbury identity, and
% the log-determinant and quadratic form are computed from the Cholesky factor
% of the precision matrix, avoiding direct inversion of an n x n matrix.
%
% Inputs:
%   Y     : data, T x n
%   a     : free elements of the lower-triangular loading matrix A
%   Sig   : idiosyncratic variances, n x 1
%   Omega : factor variances, r x 1
%
% Output:
%   loglike : log integrated likelihood

[T,n] = size(Y);
r = length(Omega);

% construct unit lower-triangular A
A = [eye(r); zeros(n-r,r)];
count_a = 0;
for ii = 2:n
    nai = min(ii-1,r);
    A(ii,1:nai) = a(count_a+1:count_a+nai);
    count_a = count_a + nai;
end

% sparse diagonal precision matrices
iSig = spdiags(1./Sig(:),0,n,n);
iOmega = spdiags(1./Omega(:),0,r,r);

% compute (A*Omega*A' + Sig)^{-1} using Woodbury identity
AiSig = A' * iSig;
iB = iSig - AiSig' * ((iOmega + AiSig*A) \ AiSig);

CiB = chol(iB,'lower'); % Cholesky factor of precision
CY = CiB' * Y';
quad = sum(CY(:).^2); % quadratic term: sum_t y_t' iB y_t
loglike = -0.5*T*n*log(2*pi) + T*sum(log(diag(CiB))) - 0.5*quad;
end
