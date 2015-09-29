function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

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

% Compute cost for given theta
% dimensions of X: m x (n + 1); dimensions of theta: (n + 1) x 1;
z = X * theta;      % dimensions of z: m x 1
g = sigmoid(z);     % compute predictions by using Sigmoid Function, dimensions of g: m x 1

t = theta(2:size(theta));      % Regularization without theta 0
reg = lambda .* sum(t .^ 2) ./ (2 .* m);
J = (sum(- y .* log(g) - (1 - y) .* log(1 - g)) / m) + reg;

% Compute partial derivatives of the Cost Function for each theta
n = size(theta, 1);
grad(1) = sum((g - y) .* X(:, 1)) / m;
for j = 2:n
    grad(j) = (sum((g - y) .* X(:, j)) / m) + (lambda .* theta(j) ./ m);
end

% =============================================================

end
