function gp_prior_paths
% GP_PRIOR_PATHS  Visualize Gaussian process priors through sample paths.
%
%   Draws a handful of sample paths from a zero-mean Gaussian process under
%   two covariance functions and plots them side by side in a 1-by-2 panel:
%
%     (left)  Brownian motion kernel      k(x,x') = min(x,x')
%             -> continuous but rough (nowhere differentiable) paths
%     (right) squared exponential kernel  k(x,x') = sf2*exp(-(x-x')^2/(2*l^2))
%             -> smooth paths
%
%   The kernel is the object that encodes prior beliefs about regularity and
%   persistence: the squared exponential produces smooth curves, while the
%   Brownian motion kernel produces jagged ones. This reproduces the
%   "Visualizing a Gaussian Process" illustration in the Gaussian process
%   regression section. Paths are drawn in grayscale so the figure reads in
%   black and white.

rng(42);                                % fix the seed for reproducible paths

x      = linspace(0,1,200)';            % input grid on [0,1] (column vector)
npaths = 5;                             % number of sample paths per panel

% --- covariance (Gram) matrices on the grid ---
K_bm   = min(x,x');                     % Brownian motion kernel
sf2    = 1;                             % signal variance of the SE kernel
lambda = 0.15;                          % length scale of the SE kernel
K_se   = sf2*exp(-(x-x').^2/(2*lambda^2));   % squared exponential kernel

% --- draw sample paths  f ~ N(0,K)  (each column is one path) ---
F_bm = gp_sample(K_bm,npaths);
F_se = gp_sample(K_se,npaths);

% --- plot the 1-by-2 panel (black-and-white, grayscale paths) ---
fs   = 14;                                       % font size
lw   = 1;                                         % line width
gray = repmat(linspace(0,0.7,npaths)',1,3);       % black -> light gray
yl   = 1.05*max(abs([F_bm(:); F_se(:)]));         % common vertical scale

fig = figure('Color','w','Position',[100 100 800 320]);

subplot(1,2,1); hold on;
for j = 1:npaths
    plot(x,F_bm(:,j),'-','Color',gray(j,:),'LineWidth',lw);
end
box off; xlim([0 1.1]); ylim([-yl yl]);
set(gca,'FontSize',fs);
xlabel('$x$','Interpreter','latex','FontSize',fs);
ylabel('$f(x)$','Interpreter','latex','FontSize',fs);
title('$k_{\mathrm{BM}}$','Interpreter','latex','FontSize',fs);

subplot(1,2,2); hold on;
for j = 1:npaths
    plot(x,F_se(:,j),'-','Color',gray(j,:),'LineWidth',lw);
end
box off; xlim([0 1.1]); ylim([-yl yl]);
set(gca,'FontSize',fs);
xlabel('$x$','Interpreter','latex','FontSize',fs);
ylabel('$f(x)$','Interpreter','latex','FontSize',fs);
title('$k_{\mathrm{SE}}$','Interpreter','latex','FontSize',fs);

% Export for the book (path relative to code/chatty/):
print(fig,'gp_prior_paths.eps','-depsc2','-painters');

end
% -------------------------------------------------------------------------
function F = gp_sample(K,npaths)
% Draw NPATHS sample paths from N(0,K) via the Cholesky factor, adding a
% small jitter to the diagonal (increased adaptively) for numerical stability.
n      = size(K,1);
jitter = 1e-12;
p      = 1;
while p > 0
    [L,p] = chol(K + jitter*eye(n),'lower');
    if p > 0, jitter = jitter*10; end   % not yet positive definite: add more
end
F = L*randn(n,npaths);
end
