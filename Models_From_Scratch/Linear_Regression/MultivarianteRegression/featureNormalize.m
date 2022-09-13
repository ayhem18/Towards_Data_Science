function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. 
% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
num_samples = size(X, 1);



mu = mu + mean(X);
m = zeros(size(mu)) + mu;

% for iter = 1:num_iters
    % theta = theta  - (alpha * X' * (X * theta - y)) / m
    % Save the cost J in every iteration    
    % J_history(iter) = computeCost(X, y, theta);
% end


for i = 1:num_samples-1
  m = vertcat(m, mu);
end 

sigma = sigma + std(X);
s = zeros(size(sigma)) + sigma

for i =1:num_samples - 1
  s = vertcat(s, sigma);
end
X_norm = (X_norm .- m) ./ s;

% ============================================================

end
