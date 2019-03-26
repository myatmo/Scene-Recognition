function [feature] = GetTinyImage(I, output_size)
%Input: I is an gray scale image, output_size=[w,h] is the size of the tiny image.
%Output: feature is the tiny image representation by vectorizing the pixel intensity in
%a column major order. The resulting size will be w×h.
%Description: You will simply resize each image to a small, fixed resolution (e.g.,
%16×16). You need to normalize the image by having zero mean and unit length. This
%is not a particularly good representation, because it discards all of the high frequency
%image content and is not especially invariant to spatial or brightness shifts.

w = output_size(1);
h = output_size(2);

% vectorization of the image in column-major order
    %J = reshape(I,[],1);

% resizing the result vector
J = imresize(I, 1/w);
whos J
J = im2double(J);
J = (J-mean(J(:))) / std(J(:));

% vectorization
feature = reshape(J,[],1);
whos feature

    %feature = reshape(J, [w ,h]);
    %whos feature
    %imshow(feature)


