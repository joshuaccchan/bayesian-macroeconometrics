function k = ldet(Omega)
% ldet.m
% Computes the log determinant of a symmetric positive definite matrix from
% its lower-triangular Cholesky factor: log|Omega| = 2*sum(log(diag(C))).
%
% Input:
%   Omega : symmetric positive definite matrix
%
% Output:
%   k : log determinant of Omega
k = 2*sum(log(diag(chol(Omega,'lower'))));
end
