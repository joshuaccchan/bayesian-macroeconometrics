function yIR = construct_IR(beta,Sig,n_hz,shock)
% construct_IR.m
% Computes the structural impulse response function of a VAR(p) by iterating
% the VAR forward under two scenarios -- one in which a structural shock hits
% at the impact horizon and one without -- and differencing the two paths.
%
% Inputs:
%   beta  : nk x 1 vector of VAR coefficients, k = 1+n*p
%   Sig   : n x n error covariance matrix
%   n_hz  : number of horizons (including impact, h = 0)
%   shock : n x 1 structural shock vector (e.g., a unit vector e_j)
%
% Output:
%   yIR : n_hz x n impulse responses; row h gives the response at horizon h-1

n = size(Sig,1);
p = (size(beta,1)/n-1)/n;
CSig = chol(Sig,'lower');

% initialize: shocked path starts at CSig*shock, baseline at 0
tmpZ1 = zeros(p,n); tmpZ = zeros(p,n);
Yt1 = CSig*shock; Yt = zeros(n,1);
yIR = zeros(n_hz,n); yIR(1,:) = Yt1';

for t = 2:n_hz
    % update the lagged values for each path
    tmpZ = [Yt'; tmpZ(1:end-1,:)];
    tmpZ1 = [Yt1'; tmpZ1(1:end-1,:)];

    % shocked path: iterate the VAR forward
    Z1 = reshape(tmpZ1',1,n*p);
    Xt1 = kron(speye(n), [1 Z1]);
    Yt1 = Xt1*beta;

    % baseline path: iterate the VAR forward
    Z = reshape(tmpZ',1,n*p);
    Xt = kron(speye(n), [1 Z]);
    Yt = Xt*beta;

    % impulse response = difference between the two paths
    yIR(t,:) = (Yt1-Yt)';
end
end
