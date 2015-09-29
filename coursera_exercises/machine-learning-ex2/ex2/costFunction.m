function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Compute cost for given theta
% dimensions of X: m x (n + 1); dimensions of theta: (n + 1) x 1;
z = X * theta;      % dimensions of z: m x 1
g = sigmoid(z);     % compute predictions by using Sigmoid Function, dimensions of g: m x 1

J = sum(- y .* log(g) - (1 - y) .* log(1 - g)) / m;

% Compute partial derivatives of the Cost Function for each theta
n = size(theta, 1);
for j = 1:n
    grad(j) = sum((g - y) .* X(:, j)) / m;
end

% =============================================================

end
