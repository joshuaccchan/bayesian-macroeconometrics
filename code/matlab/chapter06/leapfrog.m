function [thetaNew, pNew] = leapfrog(theta, p, eps, L, grad_logf)
% leapfrog.m
% Leapfrog integrator for Hamiltonian Monte Carlo with Gaussian
% momentum. Implements L leapfrog steps with step size eps to evolve
% (theta, p) along the simulated Hamiltonian trajectory. Adjacent
% half-step momentum updates are combined, so an L-step trajectory
% requires only L+1 evaluations of grad_logf rather than 2L.
%
% Inputs:
%   theta    : current position (k-by-1)
%   p        : current momentum (k-by-1)
%   eps      : step size
%   L        : number of leapfrog steps
%   grad_logf: function handle returning grad log f(theta)
%
% Outputs:
%   thetaNew : proposed position after L leapfrog steps
%   pNew     : proposed momentum after L leapfrog steps
thetaNew = theta;
pNew = p;

% initial half-step for momentum
g = grad_logf(thetaNew);
pNew = pNew + 0.5*eps*g;

% full steps
for t = 1:(L-1)
    thetaNew = thetaNew + eps*pNew;
    g = grad_logf(thetaNew);
    pNew = pNew + eps*g;
end

% final position update and half-step momentum
thetaNew = thetaNew + eps*pNew;
g = grad_logf(thetaNew);
pNew = pNew + 0.5*eps*g;
end
