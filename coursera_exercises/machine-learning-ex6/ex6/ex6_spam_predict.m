function spam = ex6_spam_predict(filename)

spam = 0;

if ~exist('filename', 'var') || isempty(filename)
    filename = 'spamSample1.txt';
end

% Load traied model, we get 'model' to predict
load('model.mat');

% Read and predict
file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x             = emailFeatures(word_indices);
p = svmPredict(model, x);

fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');

spam = p;

end