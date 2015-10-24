function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% ===========================================================
% Implement in Loop version, more readable
% ===========================================================
% m = size(X, 1);
% for i = 1:m
%     xi = X(i, :);   % 1 x n vector
%     for k = 1:K
%         uk = centroids(k, :);   % 1 x n vector
%         if k == 1
%             dist = (xi - uk) * (xi - uk)';
%             idx(i) = k;
%         else
%             new_dist = (xi - uk) * (xi - uk)';
%             if new_dist < dist
%                 dist = new_dist;
%                 idx(i) = k;
%             end
%         end
%     end
% end

% ===========================================================
% Implement in vector version, more efficient
% ===========================================================
xk = zeros(size(X, 1), K);  % m x k matrix
for i = 1:K
    xi = X - centroids(i, :);   % m x n matrix. sub each row of X to centroids(i, :).
    xi = xi .* xi;  % m x n matrix. each row now have a component of x_i^2.
    xi = sum(xi, 2);    % m x 1 vector. add each component of x_i^2 in rows. This is the distance of xi and uk.
    if i == 1
        xk = xi;
    else
        xk = [xk xi];   % append each column to xk
    end
end

% xk(i, :) have K component, each component is the distance between xi and uk
[val, idx] = min(xk, [], 2);    % return the min value and it's index of each row

% =============================================================

end

