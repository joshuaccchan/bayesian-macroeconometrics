% VAR_CSV_o.m
% Metropolis-within-Gibbs sampler for a Bayesian VAR with common stochastic
% volatility and an outlier component. The VAR coefficients A and the 
% covariance Sig have a natural conjugate prior. 
% Requires Minn_NCP.m, sample_CSV_h_ARMH.m, SV_RW_gaussian_approx.m,
% shaded_band.m, and inefficiency_factor.m (with specvar0.m).

clear; clc; rng(42);

p = 12;  % number of lags (monthly data)
nsims = 30000;
burnin = 2000;

% load monthly FRED-MD data: the 16 variables of Carriero, Clark, Marcellino
% and Mertens (2024), augmented with nine series spanning interest rates,
% money, credit, and prices for a 25-variable panel
raw = readtable('FRED-MD.csv','VariableNamingRule','preserve');
vars = {'RPI','DPCERA3M086SBEA','INDPRO','CUMFNS','UNRATE','PAYEMS', ...
        'CES0600000007','CES0600000008','WPSFD49207','PCEPI','HOUST', ...
        'S&P 500','EXUSUKx','GS5','GS10','BAAFFM', ...  % Carriero et al. (2024)
        'FEDFUNDS','TB3MS','GS1','M2SL','BUSLOANS','CPIAUCSL', ...
        'OILPRICEx','RETAILx','PERMIT'};                % additional series
data = raw{:,vars};
dates = raw{:,1};
ok = all(~isnan(data),2);  % keep fully observed months
data = data(ok,:); dates = dates(ok);
Y0 = data(1:p,:);      % pre-sample obs (initial conditions)
Y = data(p+1:end,:);   % estimation sample
dates = dates(p+1:end);
fprintf('FRED-MD: n=%d variables, sample %.2f-%.2f (%d months)\n', ...
    numel(vars), dates(1), dates(end), length(dates));
[T,n] = size(Y);
k = n*p+1;

% prior hyperparameters
kappa1 = 100;    % prior variance on intercepts
kappa2 = 0.2^2;  % overall shrinkage on lag coefficients
rw = 0;          % zero prior mean
[A0, VA, nu0, S0] = Minn_NCP(Y, Y0, p, kappa1, kappa2, rw);
iVA = sparse(1:k,1:k,1./diag(VA));
phi0 = 0.95; Vphi = 0.1^2;    % common SV: phi ~ N(phi0,Vphi) on (-1,1)
nuh0 = 20; Sh0 = 0.05*(nuh0-1); % sigh2 ~ IG(nuh0,Sh0)
p0a = 10/4; p0b = (1-1/48)*120; % outlier prob p_o ~ Beta(p0a,p0b)
ngrid = 100;                    % grid points for the U(2,10) component
o_grid = [1; linspace(2,10,ngrid)'];  % support of o_t: point mass at 1 plus grid

% construct the regressor matrix Z = [1, y_{t-1}',...,y_{t-p}']
tmpY = [Y0(end-p+1:end,:); Y];
Z = zeros(T,n*p);
for i = 1:p
    Z(:,(i-1)*n+1:i*n) = tmpY(p-i+1:end-i,:);
end
Z = [ones(T,1) Z];

% storage
store_A = zeros(k,n);
store_Sig = zeros(n,n);
store_h = zeros(nsims,T);
store_o = zeros(nsims,T);
store_theta = zeros(nsims,3);  % [phi, sigh2, po]
counth = 0; countphi = 0;

% initialize the Markov chain
phi = phi0; sigh2 = 0.1;
o = ones(T,1); o2 = o.^2;            % o2_t = o_t^2
po = 1/48;
    % initialize h with a Gaussian approximation
A_ols = (Z'*Z + iVA)\(Z'*Y);   
U_ols = Y - Z*A_ols;          
s2_init = sum((U_ols/chol(U_ols'*U_ols/T,'lower')').^2,2); 
h = SV_RW_gaussian_approx(s2_init, mean(log(s2_init)), sigh2);
h = h - mean(h);      % common SV is normalized to zero mean

disp('Starting MCMC for the VAR with common SV and outliers....');
tic;
for isim = 1:nsims+burnin
    % sample Sig and A given (h, o2) with Omega^{-1} = diag(exp(-h)./o2)
    iOm = sparse(1:T,1:T,exp(-h)./o2); 
    ZiOm = Z'*iOm;
    KA = iVA + ZiOm*Z;
    CKA = chol(KA,'lower');
    Ahat = CKA'\(CKA\(iVA*A0 + ZiOm*Y));
    Shat = S0 + A0'*iVA*A0 + Y'*iOm*Y - Ahat'*KA*Ahat;
    Shat = (Shat+Shat')/2;   % adjust for rounding errors
    Sig = iwishrnd(Shat,nu0+T);
    CSig = chol(Sig,'lower');
    A = Ahat + (CKA'\randn(k,n))*CSig';

    % sample the common log-volatility h via the Laplace-based ARMH step
    U = Y - Z*A;    % reduced-form residuals
    tmp = U/CSig';  % residuals standardized by chol(Sig)
    s2 = sum(tmp.^2,2)./o2;  % per-period sum of squares over n vars
    [h,flag] = sample_CSV_h_ARMH(s2,phi,sigh2,h,n,30);  % kappa=30 for better mixing
    counth = counth + flag;

    % sample the outlier states o_t on a grid and the outlier probability p_o
    s2 = sum(tmp.^2,2)./exp(h);   % per-period sum of squares given h
    o_lpri = log([1-po; repmat(po/ngrid,ngrid,1)]);  % prior over the grid
    for tt = 1:T
        lliket = -n*log(o_grid) - .5*s2(tt)./o_grid.^2;  % log-likelihood at each o
        lp = lliket + o_lpri;
        o_post = exp(lp - max(lp));
        o_post = o_post/sum(o_post);
        o(tt) = o_grid(find(rand<cumsum(o_post),1));   % inverse-CDF draw
    end
    n_out = sum(o>1);
    po = betarnd(p0a + n_out, p0b + T - n_out);
    o2 = o.^2;

    % sample sigh2
    eh = [h(1)*sqrt(1-phi^2); h(2:end)-phi*h(1:end-1)];
    sigh2 = 1/gamrnd(nuh0+T/2,1/(Sh0 + sum(eh.^2)/2));

    % sample phi via an independence-chain MH step
    Kphi = 1/Vphi + sum(h(1:T-1).^2)/sigh2;
    phihat = Kphi\(phi0/Vphi + h(1:T-1)'*h(2:T)/sigh2);
    phic = phihat + sqrt(Kphi)'\randn;
    gphi = @(x) -.5*log(sigh2./(1-x.^2)) - .5*(1-x.^2)/sigh2*h(1)^2;
    if abs(phic)<.9999
        if exp(gphi(phic)-gphi(phi)) > rand
            phi = phic;
            countphi = countphi + 1;
        end
    end

    if isim > burnin
        isave = isim - burnin;
        store_A = store_A + A;
        store_Sig = store_Sig + Sig;
        store_h(isave,:) = h';
        store_o(isave,:) = o';
        store_theta(isave,:) = [phi sigh2 po];
    end

    if mod(isim,1000)==0
        disp([num2str(isim) ' loops... ']);
    end
end
fprintf('MCMC takes %.1f seconds\n', toc);

% posterior summaries
% the level of h is not separately identified from Sig (only exp(h_t)*Sig is),
% so normalize each draw of h to zero mean before reporting the volatility
store_h = store_h - mean(store_h,2);
A_mean = store_A/nsims;
Sig_mean = store_Sig/nsims;
theta_mean = mean(store_theta)'; 
h_mean = mean(store_h)';
vol = exp(store_h/2);  % e^{h_t/2}: normalized uncertainty measure
vol_mean = mean(vol)';
vol_ci = quantile(vol,[.05 .95])';   % 90% credible interval
o_mean = mean(store_o)';
o_ci = quantile(store_o,[.05 .95])'; % 90% credible interval
oprob = mean(store_o>1)';            % posterior P(o_t > 1)

fprintf('posterior mean p_o: %.3f; months with P(o_t>1)>0.5: %d\n', ...
    theta_mean(3), sum(oprob>0.5));
fprintf('acceptance rate (h): %.3f\n', counth/(nsims+burnin));

% inefficiency factors (integrated autocorrelation times) of the T elements
% of the common log-volatility h, summarizing the mixing of the ARMH step.
% The spectral estimator uses a Bartlett window with truncation lag 500.
IF_h = inefficiency_factor(store_h, 500);   % store_h is nsims x T
fprintf('IF of h: mean %.1f, median %.1f\n', mean(IF_h), median(IF_h));

% common volatility e^{h_t/2} with 90% credible interval
figure('Position',[100 100 800 300]);
shaded_band(dates, vol_ci(:,1), vol_ci(:,2)); hold on;
plot(dates, vol_mean, 'k'); box off; hold off;
% title('common volatility e^{h_t/2}');
set(gca,'FontSize',16);

% outlier component o_t with 90% credible interval
figure('Position',[100 100 800 300]);
shaded_band(dates, o_ci(:,1), o_ci(:,2)); hold on;
plot(dates, o_mean, 'k'); box off; hold off;
% title('outlier component o_t');
set(gca,'FontSize',16);
