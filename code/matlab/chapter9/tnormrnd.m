function draws = tnormrnd(mu, sigma2, a, b, N)
% This funciton draws from a univariate truncated normal distribution using
% the inverse-transform method. It returns N draws from a normal 
% distribution with mean mu and variance sigma2, truncated to the interval (a,b)
%
% Inputs:
% mu: mean (scalar or N-by-1 vector)
% sigma2: variance (scalar or N-by-1 vector)
% a: lower truncation point (scalar)
% b: upper truncation point (scalar)
% N: number of draws (optional; defaults to length(mu))
%
% Output:
% draws: N-by-1 vector of truncated normal draws

    % check number of inputs
    if nargin < 4
        error('Wrong number of arguments.');
    end

    % length of mean vector
    K = length(mu);

    % if N not supplied, set equal to length(mu)
    if nargin < 5
        N = K;
    end

    % dimension check
    if ( (K ~= N || length(sigma2) ~= N) && K ~= 1 )
        error('Dimensions of mu and sigma2 must equal N.');
    end
    
    % expand scalars to vectors if necessary
    if K == 1
        mu     = repmat(mu, N, 1);
        sigma2 = repmat(sigma2, N, 1);
    end

    sigma = sqrt(sigma2);    
    u = rand(N,1);

    % compute CDF values at truncation points
    p1 = normcdf((a - mu) ./ sigma);
    p2 = normcdf((b - mu) ./ sigma);

    % apply inverse CDF transformation
    C = norminv(p1 + (p2 - p1).*u);

    % transform back to truncated normal draw
    draws = mu + sigma .* C;
end
