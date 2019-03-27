function [confusion, accuracy] = ClassifyKNN_Tiny
%Output: confusion is a 15 × 15 confusion matrix and accuracy is the accuracy of the
%testing data prediction.
%Description: You will combine GetTinyImage and PredictKNN for scene classification.
%Your goal is to achieve the accuracy >18%.

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

% loading images and building image representation
w = 16;h = 16;
output_size = [w, h];
k = 5; % number of neighbours
for i = 1:length(path_train)
    I = imread(char(scene+'\'+path_train(i)));
    feature_train = [feature_train; GetTinyImage(I, output_size)]; % GetTinyImage for each training image
end

for i = 1:length(path_test)
   I = imread(char(scene+'\'+path_test(i)));
   feature_test = [feature_test; GetTinyImage(I, output_size)]; % GetTinyImage for each test image
end

% predict label test using KNN classifier
label_test_pred = PredictKNN(feature_train, label_train, feature_test, k);
label_test_pred = string(label_test_pred); % convert cell array to string

% confusion matrix
confusion = confusionmat(label_test, label_test_pred);

% calculating accuracy
accuracy = sum(diag(confusion)) / sum(sum(confusion));

