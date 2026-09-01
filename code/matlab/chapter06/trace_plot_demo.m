clear; clc;

R = 2200;
burnin = 200;
rng(0);

% -------------------------------------------------------------------------
% (1) Well-mixed AR(1): low persistence
% -------------------------------------------------------------------------
phi1 = 0.20;  sig1 = sqrt(1 - phi1^2);
x1 = zeros(R,1);
for r = 2:R
    x1(r) = phi1*x1(r-1) + sig1*randn;
end

% -------------------------------------------------------------------------
% (2) Highly autocorrelated AR(1): high persistence
% -------------------------------------------------------------------------
phi2 = 0.98;  sig2 = sqrt(1 - phi2^2);
x2 = zeros(R,1);
for r = 2:R
    x2(r) = phi2*x2(r-1) + sig2*randn;
end

% -------------------------------------------------------------------------
% (3) Sticky chain: stays put with high probability (plateaus)
% -------------------------------------------------------------------------
p_stay = 0.97;
sig3 = 0.50;
x3 = zeros(R,1);
for r = 2:R
    if rand < p_stay
        x3(r) = x3(r-1);
    else
        x3(r) = x3(r-1) + sig3*randn;
    end
end

% Drop burn-in for display
idx = (burnin+1):R;
x1 = x1(idx); x2 = x2(idx); x3 = x3(idx);

% -------------------------------------------------------------------------
% 1x3 panel of trace plots (black and white)
% -------------------------------------------------------------------------
figure;

subplot(1,3,1);
plot(x1,'k','LineWidth', 1);
% title('Well-mixed chain');
xlabel('Iteration','FontSize',14); ylabel('$x^{(r)}$', 'FontSize',14,'Interpreter','latex');
box off;

subplot(1,3,2);
plot(x2,'k','LineWidth', 1);
% title('High autocorrelation');
xlabel('Iteration','FontSize',14); ylabel('$x^{(r)}$', 'FontSize',14,'Interpreter','latex');
box off;

subplot(1,3,3);
plot(x3,'k','LineWidth', 1);
% title('Sticky chain (stuck)');
xlabel('Iteration','FontSize',14); ylabel('$x^{(r)}$', 'FontSize',14,'Interpreter','latex');
box off;

set(gcf,'Position',[100 100 800 300]);
