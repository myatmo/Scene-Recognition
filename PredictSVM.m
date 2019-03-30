function [label_test_pred] = PredictSVM(feature_train, label_train, feature_test)
%Input: feature_train is a n_tr × d matrix where ntr is the number of training data samples and
%d is the dimension of image feature. Each row is the image feature. label_train? [1, 15] is
%a n_tr vector that specifies the label of the training data. feature_test is a n_te × d matrix
%that contains the testing features where nte is the number of testing data samples.
%Output: label_test_pred is a n_te vector that specifies the predicted label for the testing data.

lambda = 0.1 ; % Regularization parameter
maxIter = 1000 ; % Maximum number of iterations

% for each class, construct label as binary (1 vs -1) and build SVM classifier
score = [];
classes = unique(label_train); % get all the classes
for i = 1:length(classes)
    %y{i} = ismember(label_train, labels(i));
    w = 0; b = 0;
    for j = 1:length(label_train)
        if label_train(j) == classes(i)
            label{i}(j) = 1;
        else
            label{i}(j) = -1;
        end
        label{i} = im2double(label{i});
    end
    
    % training SVM
    [w, b] = vl_svmtrain(feature_train', label{i}.', lambda, 'MaxNumIterations', maxIter);
    
    % getting score of test images by each classifier
    s = w' * feature_test' + b;
    score = [score, s']; % each column is for each classifier correspond to classes
end

% getting max score from all classifiers to predict labels
label_test_pred = [];
for i = 1:length(score)
    [~,idx] = max(score(i,:));
    label_test_pred = [label_test_pred; classes(idx)];
end



