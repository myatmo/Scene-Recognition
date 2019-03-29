function [bow_feature] = ComputeBoW(feature, vocab)
%Input: feature is a set of SIFT features for one image, and vocab is visual dictionary.
%Output: bow_feature is the bag-of-words feature vector whose size is dic_size.
%Description: Give a set of SIFT features from an image, you will compute the bag-ofwords feature.
%The BoW feature is constructed by counting SIFT features that fall into each cluster of the vocabulary.
%Nearest neighbor can be used to find the closest cluster center.
%The histogram needs to be normalized such that BoW feature has a unit length.

feature = im2double(feature); % convert to double
idx = knnsearch(vocab, feature); % get index of nearest neighbour in vocab
%matches = vocab(idx,:);

% building histogram
bow_feature = zeros(size(vocab,1),1);
for i = 1:size(idx,1)
    for j = 1:size(vocab,1)
        if idx(i) == j
            bow_feature(j) = bow_feature(j) + 1; % number of occurrence in each vocab row
        end
    end
end

% normalize to unit length
bow_feature = bow_feature / norm(bow_feature);

