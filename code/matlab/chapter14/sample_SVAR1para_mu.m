function [mu, phi, sigh2] = sample_SVAR1para_mu(h, mu, phi, sigh2, ...
    mu0, Vmu, phi0, Vphi, nu_h, S_h, free_mu)
% This function jointly samples the level mu_i, the persistence phi_i,
% and the innovation variance sigh2_i of each stationary AR(1)
% log-volatility equation
%     h_{i,t} = mu_i + phi_i*(h_{i,t-1} - mu_i) + u_{i,t},
%     u_{i,t} ~ N(0, sigh2_i),
%     h_{i,1} ~ N(mu_i, sigh2_i/(1 - phi_i^2)),
% given a current draw of h. sigh2_i is drawn from its IG full
% conditional; phi_i is updated via an independence-chain
% Metropolis-Hastings step whose acceptance ratio carries the
% sqrt(1 - phi_i^2) prefactor from the stationary initial condition;
% mu_i is drawn from its Gaussian full conditional. Setting
% free_mu(i) = false fixes mu_i at zero (e.g., for factor
% log-volatilities whose scale is absorbed into the loadings).
%
% Inputs:
% h:       T x n matrix of log-volatilities
% mu:      n x 1 current levels
% phi:     n x 1 current persistence
% sigh2:   n x 1 current innovation variances
% mu0:     n x 1 prior mean of mu_i
% Vmu:     n x 1 prior variance of mu_i
% phi0:    n x 1 prior mean of phi_i
% Vphi:    n x 1 prior variance of phi_i
% nu_h:    n x 1 IG shape parameter of sigh2_i
% S_h:     n x 1 IG scale parameter of sigh2_i
% free_mu: n x 1 logical; true = sample mu_i, false = fix mu_i at 0
%
% Outputs:
% mu:      n x 1 updated levels
% phi:     n x 1 updated persistence
% sigh2:   n x 1 updated innovation variances

[T, n] = size(h);
hd = h - repmat(mu', T, 1);   % demeaned log-volatilities

% sample sigh2_i ~ IG
e_h = [hd(1, :).*sqrt(1 - phi.^2)'; ...
    hd(2:end, :) - repmat(phi', T-1, 1).*hd(1:end-1, :)];
sigh2 = 1./gamrnd(nu_h + T/2, 1./(S_h + sum(e_h.^2)'/2));

% sample phi_i via independence-chain Metropolis-Hastings
Kphi = 1./Vphi + sum(hd(1:T-1, :).^2)'./sigh2;
phi_hat = (phi0./Vphi + sum(hd(1:T-1, :).*hd(2:T, :))'./sigh2)./Kphi;
phic = phi_hat + 1./sqrt(Kphi).*randn(n, 1);
for i = 1:n
    g_phi = @(x) 0.5*log(1 - x^2) - 0.5*(1 - x^2)/sigh2(i)*hd(1, i)^2;
    if abs(phic(i)) < 0.998
        if exp(g_phi(phic(i)) - g_phi(phi(i))) > rand
            phi(i) = phic(i);
        end
    end
end

% sample mu_i from its Gaussian full conditional
for i = 1:n
    if free_mu(i)
        Kmu = 1/Vmu(i) + ((1 - phi(i)^2) + (T-1)*(1 - phi(i))^2)/sigh2(i);
        mu_hat = (mu0(i)/Vmu(i) + (1 - phi(i)^2)/sigh2(i)*h(1, i) ...
            + (1 - phi(i))/sigh2(i) ...
            *sum(h(2:end, i) - phi(i)*h(1:end-1, i)))/Kmu;
        mu(i) = mu_hat + randn/sqrt(Kmu);
    end
end
end
