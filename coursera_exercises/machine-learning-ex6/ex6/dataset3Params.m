function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

C_candidates = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_candidates = [0.01 0.03 0.1 0.3 1 3 10 30];
error_candidates = ones(size(C_candidates), size(sigma_candidates));

for i=1:size(C_candidates)
    for j=1:size(sigma_candidates)
        C = C_candidates(i);
        sigma = sigma_candidates(j);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        preds = svmPredict(model, Xval);
        error_candidates(i, j) = mean(double(preds ~= yval));
        fprintf('Error for C(%f) sigma(%f): %f\n', C, sigma, err);
    end
end

[C_min, C_idx] = min(error_candidates);
[sigma_min, sigma_idx] = min(min(error_candidates));

C = C_candidates(C_idx(sigma_idx));
sigma = sigma_candidates(sigma_idx);
% =========================================================================

end
