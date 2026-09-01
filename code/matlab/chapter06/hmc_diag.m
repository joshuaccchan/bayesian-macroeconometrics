% hmc_diag.m
% Computes MCMC diagnostics for the post burn-in HMC draws produced by
% hmc_demo.m: the inefficiency factor (integrated autocorrelation time),
% the Monte Carlo standard error of the posterior mean, and Geweke's
% convergence Z-statistic, for each coordinate of theta. Requires
% hmc_demo.m, leapfrog.m, inefficiency_factor.m, mcse.m, geweke_diag.m,
% and specvar0.m.

hmc_demo;  % obtain post burn-in draws (samples)
L = 50;    % truncation lag
IF = inefficiency_factor(samples, L);
MCSE = mcse(samples, L);
[Z, pval] = geweke_diag(samples);

fprintf('\n            theta_1   theta_2\n');
fprintf('IF        %8.2f  %8.2f\n', IF);
fprintf('MCSE      %8.4f  %8.4f\n', MCSE);
fprintf('Geweke Z  %8.2f  %8.2f\n', Z);
