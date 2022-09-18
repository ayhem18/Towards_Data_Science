
%% Initialization
clear ; close all; clc

%% ================= Part 1: Find Closest Centroids ====================
% Given a random number of centroids K and 
% a random initial set of centroids find the index 
% of the closest centroids: by calling the corresponding function.

fprintf('Finding closest centroids.\n\n');

% Load an example dataset that we will be using
load('ex7data2.mat');

% Select an initial set of centroids
K = 3; % 3 Centroids
initial_centroids = [3 3; 6 2; 8 5];

% Find the closest centroids for the examples using the
% initial_centroids
idx = findClosestCentroids(X, initial_centroids);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ===================== Part 2: Compute Means =========================
% compute the next centroids: the mean of points at each current group.

%  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K);

fprintf('Centroids computed after initial finding of closest centroids: \n')
fprintf(' %f %f \n' , centroids');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Part 3: K-Means Clustering ======================
% Use the K-Means algorithm to cluster the points in the second dataset
fprintf('\nRunning K-Means clustering on example dataset.\n\n');

% Load an example dataset
load('ex7data2.mat');

% Settings for running K-Means
K = 3;
max_iters = 10;

%% run the K-means algorithm with the same intial configuration as
%% part 2

initial_centroids = [3 3; 6 2; 8 5];

% the True: paramter displays the algorithm's progress
[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);
fprintf('\nK-Means Done.\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Part 4: first part of image compression ===============
%% In this part, the K-Means algorithm will divide the image into the 
%% most prominent n pixels (n is chosen manually). It is the first step of
%% an image compression process

fprintf('\nRunning K-Means clustering on pixels from an image.\n\n');

%  Load an image of a bird
A = double(imread('bird_small.png'));

% If imread is not functionning as desired, uncomment the next line
%   load ('bird_small.mat');

A = A / 255; % all pixels should range from 0 to 1.

% Size of the image
img_size = size(A);

% reshape the data in a way that each row represents the RBG values of a pixel.
X = reshape(A, img_size(1) * img_size(2), 3);

% Running the K-means algorithm
% trying different values of K seems like a good idea.
K = 16; 
max_iters = 20;

% random initialization
initial_centroids = kMeansInitCentroids(X, K);

% Run K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 5: Image Compression ======================
% In this part we will use the results of Part 4 to compress an image.

fprintf('\nApplying K-Means to compress an image.\n\n');

% Find closest cluster members
idx = findClosestCentroids(X, centroids);

% Roughly speaking, each pixel can be represented by the index of its closest
% centroid. This said, We can compress the image by mapping each pixel to the 
% centroid value.
X_recovered = centroids(idx,:);

% Reshape the recovered image into proper dimensions
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

% Display the original image 
subplot(1, 2, 1);
imagesc(A); 
title('Original');

% Display compressed image side by side
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));
