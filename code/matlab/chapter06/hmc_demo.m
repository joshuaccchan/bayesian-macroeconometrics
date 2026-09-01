% hmc_demo.m
% Demonstrates Hamiltonian Monte Carlo sampling from the bivariate
% target f(theta_1, theta_2) ~
%   exp(-(theta_2 - theta_1^2)^2/20 - (theta_1 - 1)^2/2).
% Generates nsim post burn-in draws using L leapfrog steps with step
% size eps and standard Gaussian momentum.
% Requires leapfrog.m.

clear; clc; close all;
rng(42);

% log target density and its gradient
logf = @(q) -0.05*(q(2) - q(1).^2).^2 ...
    - 0.5*(q(1) - 1).^2;
grad_logf = @(q) [q(1)/5*(q(2) - q(1).^2) ...
    - (q(1)-1); -(q(2) - q(1).^2)/10 ];

% HMC settings
nsim = 5000; burnin = 1000;
eps = 0.8; L = 20;

% storage and initialization
theta = randn(2,1);
samples = zeros(nsim,2);
accepts = 0;

% main HMC loop
for isim = 1:(nsim + burnin)
    p0 = randn(2,1);

    % propose using L-step leapfrog
    [thetac, pc] = ...
        leapfrog(theta, p0, eps, L, grad_logf);

    % Hamiltonians at current and proposed states
    H0 = -logf(theta)  + 0.5*(p0'*p0);
    Hc = -logf(thetac) + 0.5*(pc'*pc);

    % MH accept/reject step
    if log(rand) < -(Hc - H0)
        theta = thetac;
        if isim > burnin
            accepts = accepts + 1;
        end
    end
    if isim > burnin
        samples(isim-burnin,:) = theta';
    end
end

fprintf('HMC acceptance rate: %.3f\n', accepts/nsim);

% plot: samples and target density contours
t1 = samples(:,1);
t2 = samples(:,2);

xg = linspace(min(t1), max(t1), 300);
yg = linspace(min(t2), max(t2), 300);
[T1,T2] = meshgrid(xg,yg);

Z = exp(-(0.05*(T2 - T1.^2).^2 + 0.5*(T1 - 1).^2));
Z = Z ./ max(Z(:));

figure('Position',[100 100 400 300]);
hold on;
contour(T1,T2,Z,12,'k');
plot(t1,t2,'.k','MarkerSize',4);
hold off; box off;
xlabel('$\theta_1$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\theta_2$', 'Interpreter', 'latex', 'FontSize', 14);

