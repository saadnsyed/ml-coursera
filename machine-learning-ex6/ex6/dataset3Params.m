function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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

Cvec = [0.03; 0.1; 0.3; 1; 3; 10; 30];
Cerror = zeros(size(Cvec, 1), 1);
for i=1:size(Cvec, 1)
  model = svmTrain(X, y, Cvec(i), @(x1, x2) gaussianKernel(x1, x2, sigma));
  predictions = svmPredict(model, Xval);
  Cerror(i) = mean(double(predictions ~= yval));
endfor
[min_error, min_error_index] = min(Cerror)
C = Cvec(min_error_index);

sigmaVec = [0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmaError = zeros(size(sigmaVec, 1), 1);
for i=1:size(sigmaVec, 1)
  model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigmaVec(i)));
  predictions = svmPredict(model, Xval);
  sigmaError(i) = mean(double(predictions ~= yval));
endfor
[min_error, min_error_index] = min(sigmaError)
sigma = sigmaVec(min_error_index);


% =========================================================================

end
