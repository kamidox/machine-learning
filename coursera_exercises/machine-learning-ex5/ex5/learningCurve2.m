function [error_train, error_val] = ...
    learningCurve2(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular,
%       it returns two vectors of the same length - error_train and
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% In practice, especially for small training sets, when you plot learning curves to debug your
% algorithms, it is often helpful to average across multiple sets of randomly selected examples to
% determine the training error and cross validation error.
%
% Concretely, to determine the training error and cross validation error for i examples, you should
% first randomly select i examples from the training set and i examples from the cross validation set.
% You will then learn the param- eters θ using the randomly chosen training set and evaluate the
% parameters θ on the randomly chosen training set and cross validation set. The above steps should
% then be repeated multiple times (say 50) and the averaged error should be used to determine the
% training error and cross validation error for i examples.
%
% For this optional (ungraded) exercise, you should implement the above strategy for computing the
% learning curves. For reference, figure 10 shows the learning curve we obtained for polynomial
% regression with λ = 0.01. Your figure may differ slightly due to the random selection of examples.
%

% ---------------------- Sample Solution ----------------------

n = size(X, 2);
m_cv = size(Xval, 2);
loops = 10;
for i = 1:m
    for k = 1:loops
        sel = randperm(m);
        sel = sel(1:i);
        X_train = X(sel, :);
        y_train = y(sel);
        [theta] = trainLinearReg(X_train, y_train, lambda);

        error_train(i) = linearRegCostFunction(X_train, y_train, theta, 0);
        error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
    end
end

error_train = error_train ./ loops;
error_val = error_val ./ loops;
% -------------------------------------------------------------

% =========================================================================

end
