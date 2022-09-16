function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
    
% LearningCurve generates the train and cross validation errors needed to
% for debugging. At each iteration, the model's paramters are computed
% only with the first i samples. The i-th training error is calculated only
% on the -ith training subset while the i-th cross validation error is calculated
% on the entire cross validation set.

% Larger intervals can be used for larger datasets
%

% Number of training examples
m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);
for i= 1:m
  % findthe parameters values when considering only the i-th subset
  [theta] = trainLinearReg(X(1:i, :), y(1:i), lambda);
  % calculate the training error on the subset
  error_train(i) = linearRegCostFunction(X(1: i, :), y(1:i),theta, 0) ;
  % calculate the validation error on the entire dataset
  error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);

end
