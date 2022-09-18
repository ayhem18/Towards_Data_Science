function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

             
                               
            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


% set the matrix representing the difference between 
% predictions  and ground truth ratings
error = (R .* (X * Theta') - Y);
J = 0.5 * (sum(sum(error .^2)));

% add regularization
J = J + 0.5 * lambda * sum(sum(X .^ 2)) + 0.5 * lambda * sum(sum(Theta .^ 2));

X_grad = error * Theta;

Theta_grad = error' * X;

% add the regularization
X_grad = X_grad + lambda * X;
Theta_grad = Theta_grad + lambda * Theta;

% compose the two gradients into a single vector.
grad = [X_grad(:); Theta_grad(:)];

end
