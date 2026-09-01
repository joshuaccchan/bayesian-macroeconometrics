function f = tpdfLS(y, mu, s2, nu)
% tpdfLS.m
% Univariate Student-t density in location-scale form, evaluated at y
% with location mu, scale s2, and nu degrees of freedom. The variance is
% nu/(nu-2)*s2 for nu > 2.

    s = sqrt(s2);
    z = (y - mu) ./ s;
    % the gamma ratio is computed via gammaln: a direct gamma(.)/gamma(.)
    % overflows to Inf/Inf = NaN once nu exceeds about 343
    c = exp(gammaln((nu+1)/2) - gammaln(nu/2)) ./ (sqrt(nu*pi) * s);
    f = c .* (1 + (z.^2)/nu).^(-(nu+1)/2);
end