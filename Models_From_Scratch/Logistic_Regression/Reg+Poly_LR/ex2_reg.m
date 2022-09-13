%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the second part
%  of the exercise which covers regularization with logistic regression.
%
%  You will need to complete the following functions in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);

% Put some labels
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;

%%% The data used in this file is not linearly seperable. Thus I am using 
%%% polynomial features as well as regularization.

%% the map feature returns a new dataset with polynomial features up to the 6-th power.
X = mapFeature(X(:,1), X(:,2), 6);

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Compute and display cost and gradient
% with all-ones theta and lambda = 10
test_theta = ones(size(X,2),1);
[cost, grad] = costFunctionReg(test_theta, X, y, 10);

fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:5));

fprintf("experimenting with different values of lambda \n")
fprintf("Please enter to continue\n")

pause;

lambdas = [0 0.1 0.5 1 10 100];
best_acc = 0

for i = 1: length(lambdas)

  % Initialize fitting parameters
  initial_theta = zeros(size(X, 2), 1);

  % Set regularization parameter lambda to 1 (you should vary this)
  lambda = lambdas(i);

  % Set Options
  options = optimset('GradObj', 'on', 'MaxIter', 400);

  % Optimize
  [theta, J, exit_flag] = ...
    fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

  % Plot Boundary
  plotDecisionBoundary(theta, X, y, 6);
  hold on;
  title(sprintf('lambda = %g', lambda))

  % Labels and Legend
  xlabel('Microchip Test 1')
  ylabel('Microchip Test 2')

  legend('y = 1', 'y = 0', 'Decision boundary')
  hold off;

  % Compute accuracy on our training set
  p = predict(theta, X);

 
  acc = mean(double(p==y)) * 100;
  if (acc > best_acc)
    best_acc = acc;
    best_lambda = lambda;
  endif
  
  fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
  fprintf("That was for lambda %f\n", lambda);
  fprintf("Please enter any key to continue");
  pause;
endfor

%% as we determined the best lambda, let's determine the best
%% polynomial degree
fprintf('best regularization paramter %f\n', best_lambda)
fprintf('best training accuracy %f\n', best_acc);
fprintf("Such result is expected as the regularization increases the training error for lower generalization error")
