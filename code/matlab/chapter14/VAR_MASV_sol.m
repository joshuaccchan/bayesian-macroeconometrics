% VAR_MASV_sol.m
% Solution to the exercise "Moving Average Stochastic Volatility", part (c):
% a Bayesian VAR whose reduced-form errors follow a common MA(1) with a common
% stochastic-volatility innovation:
%     y_t   = A'x_t + eps_t,
%     eps_t = v_t + psi_1 v_{t-1},          |psi_1| < 1,
%     (v_t | Sig,h_t) ~ N(0, exp(h_t) Sig),
%     h_t   = phi h_{t-1} + u_t^h,   u_t^h ~ N(0,sigh2),  h_1 ~ N(0,sigh2/(1-phi^2)).
%
% Stacking the T observations (v_0 = 0) gives U = M_psi V, with M_psi the T x T
% unit lower-bidiagonal MA operator (1 on the diagonal, psi_1 on the first
% subdiagonal) and V the T x n whitened innovations (rows v_t'). Hence
% vec(U) ~ N(0, Sig kron Omega) with Omega = M_psi diag(exp(h)) M_psi' the
% symmetric tridiagonal (banded) matrix with
%     Omega_11   = exp(h_1),
%     Omega_tt   = exp(h_t) + psi_1^2 exp(h_{t-1}),  t >= 2,
%     Omega_{t,t-1} = psi_1 exp(h_{t-1}).
% Conditional on Omega, (A,Sig) has the natural-conjugate NIW posterior of
% Theorem 15.1 (thm:largeVAR-cond-NIW). All T x T operations use the sparse
% banded Omega and its Cholesky -- no dense T x T inverses are formed.
%
% Gibbs blocks:
%   1. (A,Sig | Y,Omega): natural-conjugate NIW.
%   2. (psi_1 | Y,A,Sig,h): random-walk Metropolis-Hastings on (-1,1). Because
%      |M_psi|^n = 1, the log-conditional is
%      -0.5 sum_t exp(-h_t) v_t(psi_1)' Sig^{-1} v_t(psi_1) + log prior(psi_1),
%      with V(psi_1) = M_psi \ U obtained by a bidiagonal solve.
%   3. (h | Y,A,Sig,psi_1,phi,sigh2): whiten V = M_psi \ U so v_t ~ N(0,e^{h_t}Sig)
%      is a common-SV structure, then the same ARMH update as VAR_CSV_o.m.
%   4. (phi,sigh2 | h): exactly as in VAR_CSV_o.m.
% The outlier component of VAR_CSV_o.m is omitted here for simplicity.
%
% Requires Minn_NCP.m, sample_CSV_h_ARMH.m, SV_RW_gaussian_approx.m,
% shaded_band.m, and inefficiency_factor.m (with specvar0.m).

clear; clc; rng(42);

p = 12;            % number of lags (monthly data)
nsims = 10000;
burnin = 2000;
sig_psi = 0.03;    % random-walk MH proposal std for psi_1

% ---- load FRED-MD data (same 25-variable panel as VAR_CSV_o.m) ----
raw = readtable('FRED-MD.csv','VariableNamingRule','preserve');
vars = {'RPI','DPCERA3M086SBEA','INDPRO','CUMFNS','UNRATE','PAYEMS', ...
        'CES0600000007','CES0600000008','WPSFD49207','PCEPI','HOUST', ...
        'S&P 500','EXUSUKx','GS5','GS10','BAAFFM', ...  % Carriero et al. (2024)
        'FEDFUNDS','TB3MS','GS1','M2SL','BUSLOANS','CPIAUCSL', ...
        'OILPRICEx','RETAILx','PERMIT'};                % additional series
data = raw{:,vars};
dates = raw{:,1};
ok = all(~isnan(data),2);
data = data(ok,:); dates = dates(ok);
Y0 = data(1:p,:);
Y = data(p+1:end,:);
dates = dates(p+1:end);
[T,n] = size(Y);
k = n*p+1;
fprintf('FRED-MD: n=%d variables, sample %.2f-%.2f (%d months)\n', ...
    numel(vars), dates(1), dates(end), T);

% ---- prior hyperparameters ----
kappa1 = 100; kappa2 = 0.2^2; rw = 0;
[A0, VA, nu0, S0] = Minn_NCP(Y, Y0, p, kappa1, kappa2, rw);
iVA = sparse(1:k,1:k,1./diag(VA));
phi0 = 0.95; Vphi = 0.1^2;      % common SV: phi ~ N(phi0,Vphi) on (-1,1)
nuh0 = 20; Sh0 = 0.05*(nuh0-1); % sigh2 ~ IG(nuh0,Sh0)
% psi_1 ~ U(-1,1): flat prior, so it drops out of the MH ratio

% ---- regressor matrix Z = [1, y_{t-1}',...,y_{t-p}'] ----
tmpY = [Y0(end-p+1:end,:); Y];
Z = zeros(T,n*p);
for i = 1:p
    Z(:,(i-1)*n+1:i*n) = tmpY(p-i+1:end-i,:);
end
Z = [ones(T,1) Z];

% ---- storage ----
store_A = zeros(k,n);
store_Sig = zeros(n,n);
store_h = zeros(nsims,T);
store_psi = zeros(nsims,1);
store_theta = zeros(nsims,2);   % [phi, sigh2]
counth = 0; countphi = 0; countpsi = 0;

% ---- initialize the Markov chain ----
phi = phi0; sigh2 = 0.1; psi_1 = 0;
A_ols = (Z'*Z + iVA)\(Z'*Y);
U_ols = Y - Z*A_ols;
s2_init = sum((U_ols/chol(U_ols'*U_ols/T,'lower')').^2,2);
h = SV_RW_gaussian_approx(s2_init, mean(log(s2_init)), sigh2);
h = h - mean(h);

% fixed index vectors for building the sparse tridiagonal Omega each sweep
ii = [ (1:T)'; (1:T-1)'; (2:T)' ];
jj = [ (1:T)'; (2:T)';   (1:T-1)' ];
% fixed subdiagonal pattern for the MA operator M_psi
Msub = sparse(2:T,1:T-1,1,T,T);

disp('Starting MCMC for the VAR with common MA(1)-SV errors....');
tic;
for isim = 1:nsims+burnin
    eh = exp(h);

    % ---- sparse tridiagonal Omega = M_psi diag(exp(h)) M_psi' ----
    md = eh; md(2:T) = eh(2:T) + psi_1^2*eh(1:T-1);   % main diagonal
    od = psi_1*eh(1:T-1);                             % first off-diagonal
    Om = sparse(ii,jj,[md; od; od],T,T);
    COm = chol(Om,'lower');           % sparse lower-bidiagonal Cholesky factor
    OmiZ = COm'\(COm\Z);              % Omega^{-1} Z via banded solves
    OmiY = COm'\(COm\Y);             % Omega^{-1} Y via banded solves

    % ---- block 1: (A,Sig | Y,Omega) natural-conjugate NIW ----
    KA = iVA + Z'*OmiZ;
    KA = (KA+KA')/2;
    CKA = chol(KA,'lower');
    Ahat = CKA'\(CKA\(iVA*A0 + Z'*OmiY));
    Shat = S0 + A0'*iVA*A0 + Y'*OmiY - Ahat'*KA*Ahat;
    Shat = (Shat+Shat')/2;
    Sig = iwishrnd(Shat,nu0+T);
    CSig = chol(Sig,'lower');
    A = Ahat + (CKA'\randn(k,n))*CSig';

    U = Y - Z*A;   % reduced-form residuals (rows eps_t')

    % ---- block 2: psi_1 via random-walk MH on (-1,1) (flat prior) ----
    Mpsi = speye(T) + psi_1*Msub;
    V = Mpsi\U;                         % whitened innovations, rows v_t'
    W = V/CSig';                        % v_t standardized by chol(Sig)
    llik = -0.5*sum(exp(-h).*sum(W.^2,2));
    psic = psi_1 + sig_psi*randn;
    if abs(psic) < 1
        Vc = (speye(T) + psic*Msub)\U;
        Wc = Vc/CSig';
        llikc = -0.5*sum(exp(-h).*sum(Wc.^2,2));
        if log(rand) < llikc - llik
            psi_1 = psic;
            countpsi = countpsi + 1;
            W = Wc;
        end
    end

    % ---- block 3: common log-volatility h via the Laplace-based ARMH ----
    s2 = sum(W.^2,2);                   % per-period sum of squares over n vars
    [h,flag] = sample_CSV_h_ARMH(s2,phi,sigh2,h,n,30);
    counth = counth + flag;
    h = h - mean(h);                    % common SV normalized to zero mean

    % ---- block 4: sigh2 and phi (exactly as in VAR_CSV_o.m) ----
    eh_ar = [h(1)*sqrt(1-phi^2); h(2:end)-phi*h(1:end-1)];
    sigh2 = 1/gamrnd(nuh0+T/2,1/(Sh0 + sum(eh_ar.^2)/2));
    Kphi = 1/Vphi + sum(h(1:T-1).^2)/sigh2;
    phihat = Kphi\(phi0/Vphi + h(1:T-1)'*h(2:T)/sigh2);
    phic = phihat + sqrt(Kphi)'\randn;
    gphi = @(x) -.5*log(sigh2./(1-x.^2)) - .5*(1-x.^2)/sigh2*h(1)^2;
    if abs(phic)<.9999
        if exp(gphi(phic)-gphi(phi)) > rand
            phi = phic; countphi = countphi + 1;
        end
    end

    if isim > burnin
        isave = isim - burnin;
        store_A = store_A + A;
        store_Sig = store_Sig + Sig;
        store_h(isave,:) = h';
        store_psi(isave) = psi_1;
        store_theta(isave,:) = [phi sigh2];
    end

    if mod(isim,1000)==0
        disp([num2str(isim) ' loops... ']);
    end
end
fprintf('MCMC takes %.1f seconds\n', toc);

% ---- posterior summaries ----
store_h = store_h - mean(store_h,2);   % level of h not separately identified
A_mean = store_A/nsims;
Sig_mean = store_Sig/nsims;
theta_mean = mean(store_theta)';
h_mean = mean(store_h)';
vol = exp(store_h/2);                   % e^{h_t/2}
vol_mean = mean(vol)';
vol_ci = quantile(vol,[.05 .95])';

psi_mean = mean(store_psi);
psi_ci = quantile(store_psi,[.05 .95]);
fprintf('psi_1 posterior mean %.4f, 90%% CI [%.4f, %.4f]\n', psi_mean, psi_ci(1), psi_ci(2));
fprintf('psi_1 MH acceptance rate: %.3f\n', countpsi/(nsims+burnin));
fprintf('acceptance rate (h): %.3f ; (phi): %.3f\n', counth/(nsims+burnin), countphi/(nsims+burnin));
fprintf('phi posterior mean %.3f ; sigh2 posterior mean %.4f\n', theta_mean(1), theta_mean(2));

IF_psi = inefficiency_factor(store_psi, 500);
IF_h = inefficiency_factor(store_h, 500);
fprintf('IF of psi_1: %.1f ; IF of h: mean %.1f, median %.1f\n', IF_psi, mean(IF_h), median(IF_h));

% ---- figures (saved as vector PDFs to the writeup figures folder) ----
figdir = 'figures';  % output folder (was an absolute path in the author's setup)
if ~exist(figdir,'dir'), mkdir(figdir); end

% (i) common volatility e^{h_t/2} with 90% credible band
fig1 = figure('Position',[100 100 800 300]);
shaded_band(dates, vol_ci(:,1), vol_ci(:,2)); hold on;
plot(dates, vol_mean, 'k'); box off; hold off;
set(gca,'FontSize',16);
exportgraphics(fig1, fullfile(figdir,'sol_largeVAR_MASV_h.pdf'), 'ContentType','vector');

% (ii) histogram of the posterior draws of psi_1
fig2 = figure('Position',[100 100 500 350]);
histogram(store_psi,50,'Normalization','pdf', ...
    'FaceColor',[.7 .7 .7],'EdgeColor','none'); box off;
xlabel('\psi_1'); set(gca,'FontSize',16);
exportgraphics(fig2, fullfile(figdir,'sol_largeVAR_MASV_psi.pdf'), 'ContentType','vector');

fprintf('figures written to %s\n', figdir);
