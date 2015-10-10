function [J grad] = nnCostFunctionOptimized(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

for i = 1:m
    % step 1: compute activations for each layer using traning example x(i)
    a1 = [1; X(i, :)'];     % add bias unit, size(a1) = (input_layer_size + 1) x 1
    z2 = Theta1 * a1;       % size(z2) = hidden_layer_size x 1
    a2 = sigmoid(z2);
    a2 = [1; a2];           % add bias unit, size(a2) = (hidden_layer_size + 1) x 1
    z3 = Theta2 * a2;       % size(z3) = num_labels x 1
    a3 = sigmoid(z3);       % h(x) = a3, size(a3) = num_labels x 1

    yi = zeros(num_labels, 1);
    yi(y(i)) = 1;

    % compute cost
    cost = sum(-yi .* log(a3) - (1 - yi) .* log(1 - a3));
    J = J + cost;

    % step 2: compute delta terms for output layer
    d3 = a3 - yi;           % size(d3) = num_labels x 1

    % step 3: compute hidden layer's delta terms using back propagation
    % size(Theta2') = (hidden_layer_size + 1) x num_labels; size(d3) = num_labels x 1
    % size(Theta2' * d3) = (hidden_layer_size + 1) x 1; size(z2) = hidden_layer_size  x 1
    d2 = (Theta2' * d3)(2:end) .* sigmoidGradient(z2);     % size(d2) = hidden_layer_size x 1

    % step 4: accumulate the gradient from this traning example
    Theta2_grad = Theta2_grad + d3 * a2';       % size(d3 * a2') = num_labels x (hidden_layer_size + 1)
    Theta1_grad = Theta1_grad + d2 * a1';       % size(d2 * a1') = hidden_layer_size x (input_layer_size + 1)
end

% step 5: divide accumulate gradient to m
Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

% compute the regularized terms of gradient
t1 = Theta1;
t1(:, 1) = 0;
Theta1_grad = Theta1_grad + (lambda / m) .* t1;

t2 = Theta2;
t2(:, 1) = 0;
Theta2_grad = Theta2_grad + (lambda / m) .* t2;

% =========================================================================
% Compute cost
J = J / m;
% compute regularization
reg = 0;
reg += sum(sum(t1 .^ 2));
reg += sum(sum(t2 .^ 2));
reg = lambda * reg / (2 * m);
J = J + reg;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
