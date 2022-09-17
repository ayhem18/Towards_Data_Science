%% Initialization
clear ; close all; clc

fprintf('Loading and Visualizing Data ...\n')

% Load from ex6data1: 
% You will have X, y in your environment
load('ex6data1.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

% Training two SVM models with different C values.

% Load from ex6data1: 
% loading X and y into the environment
load('ex6data1.mat');
Cs = [0.01, 0.1, 0.5, 1, 10, 100, 1000];

for i = 1:length(Cs)
C = Cs(i);
fprintf('\nTraining Linear SVM ...\n for C = %d', C)
linSVMTrainVis(X, y, C)

fprintf('Program paused. Press enter to continue.\n');
pause;


endfor

fprintf("We can see that for larger values of C, the model's accuracy.\n");
fprintf("increases as C represents the penalty of misclassified examples.\n");
fprintf("However, the boundary is less and less natural to the dataset\n");
fprintf("because of the outlier example\n");

%% Loading and Plotting the 2nd dataset
fprintf('Loading and Visualizing Data ...\n')
load('ex6data2.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

% As we can see the data is not linearly seperable, which requires
% a more sophisticated kernel such as the Gaussian Kernel also known as
% RBF kernel: 
C_s = [0.1 1 10];
sigma_s = [0.01, 0.05, 1];

C = 1;
sigma = 0.1;
RBFSVMTrainVis(X, y, C, sigma)

fprintf('Program paused. Press enter to continue.\n');
pause;

% Loading the 3rd dataset
fprintf('Loading and Visualizing Data ...\n')
load('ex6data3.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

% use grid search to find the optimal pair (C, sigma) with the 3rd dataset
[C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;