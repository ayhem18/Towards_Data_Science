function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)

% This function offers a visualization tool to choose the best 
% regularization parameter.

% a set of possible values of lambda
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);


for i=1:length(lambda_vec)
  l = lambda_vec(i);
  theta = trainLinearReg(X, y, l);
  error_train(i) = linearRegCostFunction(X, y,theta, 0); 
  error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
endfor


end
