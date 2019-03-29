% Load training and testing images from scene_classification_data folder
scene = "scene_classification_data";
rand_train = [];
rand_test = [];

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
%labels = unique(label_train)
%length(labels)

training_image_cell = {};
% loading images
for i = 1:3%length(path_train)
    I = imread(char(scene+'\'+path_train(i)));
    training_image_cell{i} = I;
    %figure;
    %imshow(training_image_cell{i})
end

dic_size = 50;
%vocab = BuildVisualDictionary(training_image_cell, dic_size);

%imshow(training_image_cell{1});
%for i = 1:length(path_test)
   %I = imread(char(scene+'\'+path_test(i)));
%end
I = imread(char(scene+'\'+path_test(1)));
I = im2single(I);
[f, d] = vl_dsift(I, 'size', 6);
feature = d';
%whos feature
%bow_feature = ComputeBoW(feature, vocab);


