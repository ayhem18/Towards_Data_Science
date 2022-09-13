function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


%%% NO regularization

% calculating the cost
h = sigmoid(X * theta);
one_J = y .* log(h) + (1 - y) .* log(1 - h);
J = -sum(one_J) / m;

% calculating the gradient
grad = (1 / m) * X' * (h - y);

%%% Adding regularization

J = J + lambda * theta(2:end)' * theta(2:end) / (2 * m);

filter = ones(size(theta));
filter(1) = 0;
grad = grad + (lambda / m) * theta .* filter;

% =============================================================








% =============================================================

grad = grad(:);

end
