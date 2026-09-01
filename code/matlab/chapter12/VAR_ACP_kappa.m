% VAR_ACP_kappa.m
% Selects the own-lag (kappa2) and other-lag (kappa3) shrinkage
% hyperparameters of the asymmetric conjugate prior for the four-variable 
% VAR(7) on macro4_Q by maximizing the closed-form marginal likelihood over
% a grid, and plots the normalized marginal-likelihood contours over the 
% two hyperparameters.
%
% Requires: ml_VAR_ACP.m

clear; clc;
rng(42); % for reproducibility

% load data
data = readmatrix('macro4_Q.csv','NumHeaderLines',1);
data = data(:,2:end);  % drop date column -> n = 4 variables
p  = 7;
Y0 = data(1:8,:);      % pre-sample (initial conditions)
Y  = data(9:end,:);    % estimation sample 1962Q1-2019Q4
[T,n] = size(Y);
kappa1 = 100;          % intercept: weak shrinkage

% residual variances from univariate AR(p) models
s2 = zeros(n,1);  tmpY = [Y0(end-p+1:end,:); Y];
for i = 1:n
    Xi = ones(T,1);
    for l = 1:p, Xi = [Xi tmpY(p-l+1:end-l,i)]; end %#ok<AGROW>
    e = Y(:,i) - Xi*(Xi\Y(:,i));  s2(i) = e'*e/(T-size(Xi,2));
end

% evaluate the log marginal likelihood over a grid of (kappa2, kappa3)
k2g = linspace(0.005,0.60,90);        % own-lag shrinkage
k3g = linspace(0.0005,0.10,90);       % other-lag shrinkage
LML = zeros(numel(k3g),numel(k2g));
for a = 1:numel(k2g)
    for b = 1:numel(k3g)
        LML(b,a) = ml_VAR_ACP(Y,Y0,p,[kappa1 k2g(a) k3g(b)],s2);
    end
end
dens = exp(LML - max(LML(:)));  % normalized surface (flat prior), max = 1

% locate the maximizer and the best symmetric (kappa2 = kappa3) value
[~,ia] = max(max(LML,[],1)); [~,ib] = max(LML(:,ia));
fsym = @(lk) -ml_VAR_ACP(Y,Y0,p,[kappa1 exp(lk) exp(lk)],s2);
ks = exp(fminbnd(fsym,log(1e-4),log(1)));
fprintf('asymmetric optimum (own,other) = (%.3f, %.4f)\n',k2g(ia),k3g(ib));
fprintf('best symmetric = %.3f;   log-ML gain over best symmetric = %.2f\n',...
    ks, max(LML(:))+fsym(log(ks)));

% plot the marginal-likelihood contours with the key points marked
figure('Position',[200 200 560 420]); hold on
contour(k2g,k3g,dens,12,'k');      % density contours
plot([0 0.1],[0 0.1],'--','Color',[0.5 0.5 0.5],'LineWidth',1.3);  % symmetric restriction
plot(0.04,0.04,'ok','MarkerFaceColor','w','MarkerSize',9,'LineWidth',1.2);  % natural conjugate, 0.2^2
plot(k2g(ia),k3g(ib),'pk','MarkerFaceColor','k','MarkerSize',14);  % maximizer
hold off; box off
xlabel('$\kappa_2$ (own lags)','Interpreter','latex','FontSize',14);
ylabel('$\kappa_3$ (other lags)','Interpreter','latex','FontSize',14);
xlim([0 0.6]); ylim([0 0.1]);
print(gcf,'VAR_ACP_kappa','-depsc2','-painters');
