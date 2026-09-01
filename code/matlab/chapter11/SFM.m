% SFM.m
% Gibbs sampler for the static factor model fitted to daily exchange-rate
% returns on nine currencies. The model is
%   y_t = A f_t + eps_t,   eps_t ~ N(0, Sigma),   f_t ~ N(0, Omega),
% where A is n x r, lower triangular with ones on the diagonal, and Sigma and
% Omega are diagonal. The 3-block Gibbs sampler draws the factors f, the free
% loadings a, and the variances (Sigma, Omega). After sampling, it estimates
% the log marginal likelihood by the cross-entropy method (SFM_CE.m) and the
% variance decomposition (vardec_SFM.m). Set r to the desired number of factors.
%
% Requires: SFM_CE.m, vardec_SFM.m, logintlike_SFM.m, lmvnpdf.m, ligampdf.m

clear; clc;
rng(42); % for reproducibility
nsim = 20000; burnin = 1000;
R = 10000; % number of importance draws for the cross-entropy method
r = 3; % # of factors;
data = readmatrix('daily_fx.csv');
returns = 100*log(data(2:end,:)./data(1:end-1,:));
Y = returns;
    
[T,n] = size(Y);
y = reshape(Y',T*n,1);
na = n*r - r*(r+1)/2;

% storage
store_a = zeros(nsim,na);
store_f = zeros(nsim,T*r);
store_sig2 = zeros(nsim,n);
store_omega2 = zeros(nsim,r);

% prior hyperparameters
a0 = 0; Va = 1; % aij iid N(a0,Va)
nusig2 = 3; Ssig2 = 1*(nusig2-1)*ones(n,1);
nuomega2 = 3; Somega2 = 1*(nuomega2-1)*ones(r,1);
prior = @(ax,s,o) -na/2*log(2*pi*Va) -.5*sum((ax-a0).^2/Va) ...
    +sum(ligampdf(s,nusig2,Ssig2)) + sum(ligampdf(o,nuomega2,Somega2));    

% initialize
varY = var(Y)';
sig2 = varY/2; % diagonal elements of Sigma
omega2 = mean(varY)/2*ones(r,1); % diagonal elements of Omega
a = randn(na,1);
A = [eye(r);zeros(n-r,r)]; 
count_a = 0;
for ii=2:n
    nai = min(ii-1,r);
    A(ii,1:nai) = a(count_a+1:count_a+nai);
    count_a = count_a + nai;
end

tStart = tic;
for isim = 1:nsim+burnin    
    % sample f
    AiSig = A'*sparse(1:n,1:n,1./sig2);
    Kf = sparse(1:T*r,1:T*r,repmat(1./omega2,T,1)) + kron(speye(T), AiSig*A); 
    fhat = Kf\(kron(speye(T), AiSig)*y);
    f = fhat + chol(Kf,'lower')' \ randn(T*r,1);
    F = reshape(f,r,T)'; % T x r - tth row is f_t
    
    % sample a
    count_a = 0;
    for i = 2:n
        if i<=r % # of elements in ai
            nai = i-1;
        else
            nai = r;
        end
        Z = F(:,1:nai);
        Kai = sparse(1:nai,1:nai,1/Va*ones(nai,1)) + Z'*Z/sig2(i);
        if i<=r
            ai_hat = Kai\(a0/Va + Z'*(Y(:,i)-F(:,i))/sig2(i));
        else
            ai_hat = Kai\(a0/Va + Z'*Y(:,i)/sig2(i));
        end
        ai = ai_hat + chol(Kai,'lower')'\randn(nai,1);
        A(i,1:nai) = ai;
        a(count_a+1:count_a+nai) = ai; 
        count_a = count_a + nai;
    end    
 
    % sample sig2 and omega2
    E = Y - F*A';
    sig2 = 1./gamrnd(nusig2+T/2,1./(Ssig2 + sum(E.^2)'/2));    
    omega2 = 1./gamrnd(nuomega2+T/2, 1./(Somega2 + sum(F.^2)'/2));
    
    if mod(isim,5000) == 0
        fprintf('Iteration %d of %d (%.1f%%), elapsed time: %.1f seconds\n', ...
            isim, nsim+burnin, 100*isim/(nsim+burnin), toc(tStart));
    end
    
    if isim>burnin
        i = isim-burnin;        
        store_a(i,:) = a';
        store_f(i,:) = f';              
        store_sig2(i,:) = sig2';
        store_omega2(i,:) = omega2';
    end
    
end
a_mean = mean(store_a)'; astd = std(store_a)';
f_mean = reshape(mean(store_f),r,T)';
sig2_mean = mean(store_sig2)'; 
omega2_mean = mean(store_omega2)'; 

[logml_CE,logml_std] = SFM_CE(store_a,store_sig2,store_omega2,Y,prior,R);
fprintf('Log marginal likelihood = %.1f (s.e. = %.2f)\n', logml_CE, logml_std);

[vd_mean, sys_mean, idio_mean] = vardec_SFM(store_a,store_sig2,store_omega2);

varnames = {'AUD','CAD','EUR','JPY','CHF','GBP','KRW','NZD','TWD'};

disp('Systematic and idiosyncratic variance shares:')
for i = 1:n
    fprintf('%s: systematic = %.2f, idiosyncratic = %.2f\n', ...
        varnames{i}, sys_mean(i), idio_mean(i));
end



 