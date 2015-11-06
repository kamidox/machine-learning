function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
%

% ==================================================================
% Native implementations
% ==================================================================
% for i = 1:num_movies
%     for j = 1:num_users
%         if R(i, j) == 1
%             J = J + ((Theta(j, :) * (X(i, :)') - Y(i, j)) .^ 2);
%         end
%     end
% end
% J = J ./ 2;
% ==================================================================
% Vectorization implementations
% ==================================================================

J = sum(sum(R .* ((X * Theta' - Y) .^ 2))) ./ 2;

% ==================================================================
% Compute collaborative filtering gradient
% ==================================================================
% ==================================================================
% Native implementations
% ==================================================================
% for i = 1:num_movies
%     for j = 1:num_users
%         if R(i, j) == 1
%             delta = Theta(j, :) * X(i, :)' - Y(i, j);
%             X_grad(i, :) = X_grad(i, :) + delta .* Theta(j, :);
%         end
%     end
% end
%
% for j = 1:num_users
%     for i = 1:num_movies
%         if R(i, j) == 1
%             delta = Theta(j, :) * X(i, :)' - Y(i, j);
%             Theta_grad(j, :) = Theta_grad(j, :) + delta .* X(i, :);
%         end
%     end
% end
% ==================================================================
% Vectorization implementations
% ==================================================================

for i = 1:num_movies
    idx = find(R(i, :) == 1);       % a list of all the users that have rated movie i
    Theta_temp = Theta(idx, :);     % num_started_users x num_features
    Y_temp = Y(i, idx);             % 1 x num_started_users
    X_grad(i, :) = (X(i, :) * Theta_temp' - Y_temp) * Theta_temp;
end

for j = 1:num_users
    idx = find(R(:, j) == 1);       % a list of all movies have rated by user j
    X_temp = X(idx, :);             % set of movies which have rated by user j.
                                    % num_rated_movies x num_features
    Y_temp = Y(idx, j)';            % set of movies which have rated by user j.
                                    % 1 x num_started_movies
    Theta_grad(j, :) = (Theta(j, :) * X_temp' - Y_temp) * X_temp;
end

% ==================================================================
% Regularized cost function
% ==================================================================

J = J + lambda .* sum(sum(Theta .^ 2)) ./ 2 + lambda .* sum(sum(X .^ 2)) ./ 2;

% ==================================================================
% Regularized gradient
% ==================================================================
X_grad = X_grad + lambda .* X;
Theta_grad = Theta_grad + lambda .* Theta;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
