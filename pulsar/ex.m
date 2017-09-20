%setenv("GNUTERM","qt")

%% Initialization
clear ; close all; clc

% Load Training Data from HTRU_2.mat
fprintf('Loading and Visualizing Data ...\n');
fprintf('Program paused. Hit enter to continue ...\n');
pause;

# Load the data set
load("HTRU_2.mat");
y_train = y_train';
y_cv = y_cv';

input_layer_size = 8;
hidden_layer_size = 25;
num_labels = 1;

# Define initial thetas with non-zero values
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
fprintf('\nTraining Neural Network... \n');

% Define options for fmincg algo
options = optimset('MaxIter', 20);

%  Define lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
               input_layer_size, ...
               hidden_layer_size, ...
               num_labels, X_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

% Check for accuracy, precision and recall on cv set
pred = predict(Theta1, Theta2, X_cv);

fprintf('\nCV setAccuracy: %f\n', mean(double(pred == y_cv)) * 100);











































