function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% LIMITATIONS: The dimensions of matrix should not large than 2, or this will not work

xdim = size(z, 1);
ydim = size(z, 2);

for i = 1:xdim
    for j = 1:ydim
        g(i, j) = (1 + e ** (-z(i, j))) ** (-1);
    end
% =============================================================

end
