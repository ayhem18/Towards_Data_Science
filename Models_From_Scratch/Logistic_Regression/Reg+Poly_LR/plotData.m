function plotData(X, y)
  
%PLOTDATA Plots the data points X and y into a new figure 
% where positive samplesare associated with "+"
% and negative ones are represented with "o"

% X is assumed to be of the shape (X, 2):  handling at most 2 features
% Create New Figure
figure; hold on;

% finding the indices of the positive and negative samples
pos = find(y==1); neg = find(y == 0);
% Plot Examples accordingly

plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...
'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);


hold off;

end
