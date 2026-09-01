function X = SURform2(Z, n)
% SURform2.m
% Constructs the seemingly unrelated regression (SUR)
% design matrix for an n-equation system with shared regressors.
% Given a T x k matrix Z whose t-th row z_t' contains the regressors
% at time t, it returns the (n*T) x (n*k) sparse matrix whose t-th
% block of n rows is I_n kron z_t', so that
%     y = X * beta,
% where y = vec(Y'), beta = vec(A), and the system Y = Z*A has
% Y (T x n) stacking the n dependent variables in columns and A
% (k x n) the regression coefficients (column j = equation j's k
% coefficients).
%
% Inputs:
% Z:        T x k matrix of regressors (row t is x_t')
% n:        scalar; number of equations in the SUR system
%
% Outputs:
% X:        (n*T) x (n*k) sparse SUR design matrix

    repZ = kron(Z,ones(n,1));
    [r,c] = size(Z);
    idi = kron((1:r*n)',ones(c,1));
    idj = repmat((1:n*c)',r,1);
    X = sparse(idi,idj,reshape(repZ',n*r*c,1));
end
