function [vocab] = BuildVisualDictionary(training_image_cell, dic_size)
%Input: training_image_cell is a set of training images and dic_size is the size of
%the dictionary (the number of visual words).
%Output: vocab lists the quantized visual words whose size is dic_size×128.
%Description: Given a set of training images, you will build a visual dictionary made
%of quantized SIFT features. You may start dic_size=50. You can use the following
%built-in functions: vl_dsift from VLFeat, kmeans from MATLAB toolbox.

% Visual Dictionary Building Algorithm
binSize = 4;
step = 10;
N_images = length(training_image_cell);
disp(N_images);
N_each = ceil(10000/N_images);
disp(N_each);
descriptors = zeros(128, N_images * N_each);
count = 0;
for i = 1:N_images
    %1: For each image, compute dense SIFT over regular grid
    %Is = imgaussfilt(I, 4); % gaussian smoothing
    I = im2single(training_image_cell{i}); % convert to single
    [~, features] = vl_dsift(I, 'Step', step); % f: 2 x num_of_keypoints, d: 128 x num_of_keypoints
    
    %keyboard
    whos features
    %features = features';
    count = count + 1
    %2: Build a pool of SIFT features from all training images
    descriptors(:,N_each * (i-1) + 1 : N_each * i) = features(:,1:N_each);
    
    %descriptors(:,N_each * (i-1) + 1 : N_each * i) = features(:,1:N_each);
end

%3: Find cluster centers from the SIFT pool using kmeans algorithms and
%   return the cluster centers as vocab.
descriptors = single(descriptors');
%keyboard
gpuDesc = gpuArray(descriptors);
[~,centers]=kmeans(gpuDesc,dic_size,'MaxIter',10000);
%[~, centers] = kmeans(descriptors, dic_size);
vocab = single(centers);


