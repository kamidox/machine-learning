function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% According to Forward Propagation
% a2 = g(z2}), while z2 = X * theta1'
% size(X) = m x n; size(theta1) = n2 x n; size(z2) = m x n2; size(a2) = m x n2
% m is the number of prediction examples; n2 is the unit number of hidden layer, which is 25

% Add bias unit which always outputs +1 to input layer
X = [ones(m, 1) X];
a2 = sigmoid(X * Theta1');

% According to Forward Propagation
% a3 = g(z3}), while z3 = a2 * theta2'
% size(a2) = m x n2; size(theta2) = n3 x n2; size(z3) = m x n3; size(a3) = m x n3
% m is the number of prediction examples; n2 is the unit number of hidden layer, which is 25;
% n3 is the unit number of output layer, which is 10;

% Add bias unit which always outputs +1 to hidden layer
a2 = [ones(m, 1) a2];
a3 = sigmoid(a2 * Theta2');
% v is the max value in each row, size(v) = m x 1;
% p is the index of max value in each row, size(p) = m x 1;
[v, p] = max(a3, [], 2);

% =========================================================================

end
