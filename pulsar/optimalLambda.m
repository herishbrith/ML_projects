% This file corresponds to the technique that is required to find the optimum
% value of lambda by processing the cost function on various values

function [optimal_lambda] = optimalLambda(X, y, X_cv, y_cv, initialTheta, ...
input_layer_size, hidden_layer_size, num_labels)

optimal_lambda = 1;
accuracy = 100;
lambdaArray = [0, 0.00001, 0.00002, 0.00004, 0.00008, 0.00016, ...
			0.00032, 0.00064, 0.00128, 0.00512, 0.01024];

for lambda = lambdaArray
	[training_cost grad] = nnCostFunction(initialTheta, ...
			               input_layer_size, ...
			               hidden_layer_size, ...
			               num_labels, ...
			               X, y, lambda);
	Theta1 = reshape(grad(1:hidden_layer_size * (input_layer_size + 1)), ...
             hidden_layer_size, (input_layer_size + 1));
	Theta2 = reshape(grad((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
             num_labels, (hidden_layer_size + 1));
	pred = predict(Theta1, Theta2, X_cv);
	currentAccuracy = mean(double(pred==y_cv));
	fprintf("Training Cost: %f\t CV Accuracy: %f\n", ...
		training_cost, currentAccuracy);

	if currentAccuracy < accuracy
		optimal_lambda = lambda;
		accuracy = currentAccuracy;
	end
end

fprintf("Optimum value of lambda comes out to be: %f", optimal_lambda);
end
