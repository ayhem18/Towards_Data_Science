%% Initialization
clear ; close all; clc

%% ==================== Part 1: Email Preprocessing ====================
% in this part I convert an email into a vector of features using the
% precoess email function.

file_name = 'emailSample1.txt'

fprintf('\nPreprocessing sample email (emailSample1.txt)\n');

% Extract Features
file_contents = readFile(file_name);
word_indices  = processEmail(file_contents);


% Print Stats
fprintf('The following vector represents a mathematical representation of ')
fprintf("the email considered: \n");
fprintf(' %d', word_indices);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ==================== Part 2: Feature Extraction ====================
% converting the numerical representation of the email to a vector 
% representing whether a word in the vocabulary list is present in email or not

fprintf('\nExtracting features from email\n');

% Extract Features
file_contents = readFile(file_name);
word_indices  = processEmail(file_contents);
features      = emailFeatures(word_indices);

% Print Stats
fprintf('Length of feature vector: %d\n', length(features));
fprintf('Number of non-zero entries: %d\n', sum(features > 0));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 3: Train Linear SVM for Spam Classification ========
% Load the training data: X, y
load('spamTrain.mat');

fprintf('\nTraining Linear SVM (Spam Classification)\n')

C = 0.1;
model = svmTrain(X, y, C, @linearKernel);

p = svmPredict(model, X);

fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

%% =================== Part 4: Test Spam Classification ================

% Loading the test dataset: X, y
load('spamTest.mat');

fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')
p = svmPredict(model, Xtest);

fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);
pause;


%% ================= Part 5: Top Predictors of Spam ====================
% The features fed to the model are vector representing the presence of
% certain words in the vocabulary list. In this part, I will display the most
% informative words according to our linear SVM classifier: (the higher the
% coefficient / weight associated with a feature, the more informative it is)
% THIS IS ONLY TRUE BECAUSE THE DATA IS NORMALIZED (binary features)


% Sort the weights and obtin the vocabulary list
[weight, idx] = sort(model.w, 'descend');
vocabList = getVocabList();

fprintf('\nTop predictors of spam: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end

fprintf('\n\n');
