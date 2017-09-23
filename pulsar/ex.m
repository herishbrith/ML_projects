setenv("GNUTERM","qt")

%% Initialization
clear ; close all; clc

% Load Training Data from HTRU_2.mat
fprintf('Loading and Visualizing Data ...\n');
fprintf('Program paused. Hit enter to continue ...\n');
pause;

# Load the data set
load("HTRU_2.mat");
y_train = y_train';
y_test = y_test';

% Set value of reduced k-dimensons
reducedDim = 3;

% Apply feature normalization and add polynomial features to training dataset
X_train_poly = polyFeatures(X_train, 5);
[X_train_poly, mu, sigma] = featureNormalize(X_train_poly);

% Convert n-dimensions to k-dimensons using PCA, for us k is 2
X_train_2D = convertNToK(X_train_poly, reducedDim);
X_train_2D = X_train_poly;

%plot2DData(X_train_2D, y_train);
fprintf('Program paused. Hit enter to continue ...\n');
pause;

K = 2;
max_iters = 1;

% When using K-Means, it is important the initialize the centroids
% randomly, but for our case, since the solution is binary, we have
% the privilege to assign each of the centroids belonging to
% different classes
%initial_centroids = kMeansInitCentroids(X_train_2D, K);
initial_centroids = [X_train_2D(y_train==0,:)(1,:); ...
	X_train_2D(y_train==1,:)(1,:)];

% Run K-Means
[centroids, idx] = runkMeans(X_train_2D, ...
	initial_centroids, max_iters, false);

% Apply feature normalization and add polynomial features to test dataset
X_test_poly = polyFeatures(X_test, 5);
[X_test_poly, mu, sigma] = featureNormalize(X_test_poly);

% Convert n-dimensions to k-dimensons using PCA, for us k is 2
X_test_2D = convertNToK(X_test_poly, reducedDim);
X_test_2D = X_test_poly;

%plot2DData(X_test_2D(1:1000,:), y_test(1:1000), centroids);
fprintf('Program paused. Hit enter to continue ...\n');
pause;

% Predict the classes of test dataset
pred = findClosestCentroids(X_test_2D, centroids);
pred(pred==1) = 0;
pred(pred==2) = 1;

pred(500:525,:), y_test(500:525,:)

% Check for accuracy, precision and recall on cv set
fprintf('\nCV setAccuracy: %f\n', mean(double(pred == y_test)) * 100);











































