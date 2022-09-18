function p = multivariateGaussian(X, mu, Sigma2)
%% Using the built-function bsxfun, this function returns the
%% probability density function given, the values mu, Sigma2
%% If sigma2 is a matrix, it is treated as covariance matrix
%% and the returned function is a multivariate PDF.
%% otherwise, each value represents the variance of the corresponding
%% feature (column) in X.

k = length(mu);

if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1)
    Sigma2 = diag(Sigma2);
end

X = bsxfun(@minus, X, mu(:)');
p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * ...
    exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));

end