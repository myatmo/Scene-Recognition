function [label_test_pred] = PredictKNN(feature_train, label_train, feature_test, k)
%Input: feature_train is a n_tr × d matrix where n_tr is the number of training data samples
%and d is the dimension of image feature, e.g., 265 for 16×16 tiny image representation.
%Each row is the image feature. label_train? [1, 15] is a n_tr vector that
%specifies the label of the training data. feature_test is a n_te × d matrix that contains
%the testing features where n_te is the number of testing data samples. k is the number
%of neighbors for label prediction.
%Output: label_test_pred is a n_te vector that specifies the predicted label for the
%testing data.
%Description: You will use a k-nearest neighbor classifier to predict the label of the
%testing data.

Mdl = fitcknn(feature_train, label_train, 'NumNeighbors', k, 'Standardize', 1);
label_test_pred = predict(Mdl,feature_test);