function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples


% calculate unregularized loss cost function
J = 0;
error = X * theta - y;
J = (error' *  error) / (2 * m);

% add regularization
J = J + lambda * theta(2:end)' * theta(2:end) / (2 * m);

% calculate gradient with no regularization
grad = zeros(size(theta));
grad = X' * (X * theta - y) / m;

% creating a filter column vector to multiply by theta element-wise
filter = ones(size(theta));
filter(1) = 0;
grad = grad + (lambda / m) * theta .* filter;

end
