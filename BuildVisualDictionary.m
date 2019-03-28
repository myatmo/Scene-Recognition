function [vocab] = BuildVisualDictionary(training_image_cell, dic_size)
%Input: training_image_cell is a set of training images and dic_size is the size of
%the dictionary (the number of visual words).
%Output: vocab lists the quantized visual words whose size is dic_size×128.
%Description: Given a set of training images, you will build a visual dictionary made
%of quantized SIFT features. You may start dic_size=50. You can use the following
%built-in functions: vl_dsift from VLFeat, kmeans from MATLAB toolbox.

% Visual Dictionary Building Algorithm
binSize = 6;
des_list = [];
for i = 1:length(training_image_cell)
    %1: For each image, compute dense SIFT over regular grid
    I = training_image_cell{i};
    %Is = imgaussfilt(I, 4); % gaussian smoothing
    I = im2single(I); % convert to single
    [f, d] = vl_dsift(I, 'size', binSize); % f: 2 x num_of_keypoints, d: 128 x num_of_keypoints
    
    %2: Build a pool of SIFT features from all training images
    des_list = [des_list; d'];
end

des_list = im2double(des_list); % convert to double

%3: Find cluster centers from the SIFT pool using kmeans algorithms and
%   return the cluster centers as vocab.
[idx, vocab] = kmeans(des_list, dic_size);
whos idx
whos vocab
