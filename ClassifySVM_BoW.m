function [confusion, accuracy] = ClassifySVM_BoW
%Output: confusion is a 15 × 15 confusion matrix and accuracy is the accuracy of the
%testing data prediction.
%Description: Given BoW features, you will combine BuildVisualDictionary, ComputeBoW,
%PredictSVM for scene classification. Your goal is to achieve the accuracy >60%.

% Load training and testing images from scene_classification_data folder
scene = "scene_classification_data";
rand_train = [];
rand_test = [];
feature_train = [];
feature_test = [];

% reading train.txt file
file_train = scene + "/train.txt";
f = fopen(file_train);
line = fgetl(f);
while ischar(line)
    rand_train = [rand_train; string(line)];
    line = fgetl(f);
end
fclose(f);

% reading test.txt file
file_test = scene + "/test.txt";
f = fopen(file_test);
line = fgetl(f);
while ischar(line)
   rand_test = [rand_test; string(line)];
   line = fgetl(f);
end
fclose(f);

% loading training and testing set
rand_train = rand_train(randperm(length(rand_train))); % randomize the training set
train_name = split(rand_train);
label_train = train_name(:,1); % label_train contains all the training labels
path_train = train_name(:,2); % path of images to get feature_train associated with label_train

rand_test = rand_test(randperm(length(rand_test))); % randomize the test set
test_name = split(rand_test);
label_test = test_name(:,1); % label_test contains all the test labels
path_test = test_name(:,2); % path of images to get feature_test associated with label_test

% loading training images
training_image_cell = {};
for i = 1:length(path_train)
    I = imread(char(scene+'\'+path_train(i)));
    I = imresize(I, [200,200]); % not all images have same dimension so that making sure all images have same dimension
    training_image_cell{i} = I;
end

% BuildVisualDictionary
dic_size = 50;
%vocab = BuildVisualDictionary(training_image_cell, dic_size);

% ComputeBoW for each image
step = 10;
for i = 1:length(path_train)
    I = imread(char(scene+'\'+path_train(i)));
    I = imresize(I, [200,200]); % not all images have same dimension so that making sure all images have same dimension
    I = im2single(I);
    [~, feature] = vl_dsift(I, 'Step', step);
    bow_feature = ComputeBoW(feature', vocab);
    feature_train = [feature_train; bow_feature'];
end

% loading test images and computing BoW
for i = 1:length(path_test)
   I = imread(char(scene+'\'+path_test(i)));
   I = imresize(I, [200,200]); % not all images have same dimension so that making sure all images have same dimension
   I = im2single(I);
   [~, feature] = vl_dsift(I, 'Step', step);
   bow_feature = ComputeBoW(feature', vocab);
   feature_test = [feature_test; bow_feature'];
end

% predict label test using SVM
label_test_pred = PredictSVM(feature_train, label_train, feature_test);

% confusion matrix
confusion = confusionmat(label_test, label_test_pred);

% calculating accuracy
accuracy = sum(diag(confusion)) / sum(sum(confusion));


