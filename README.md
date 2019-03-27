# Scene-Recognition
Tiny image and bag-of-word visual vocabulary

The scene classification dataset consists of 15 scene categories including office, kitchen, and forest.
This system will compute a set of image representations (tiny image and bag-of-word visual vocabulary) and predict the category of each testing image using the classifiers (k-nearest neighbor and SVM) built on the training data

You can download the training and testing data from here:
http://www.cs.umn.edu/~hspark/csci5561/scene_classification_data.zip

Output example of Confusion Matrix and Accuracy from for ClassifyKNN_Tiny:

confusion =

    15     1     2     2    16    10     1     4     1     6     3     2    20    17     0
     2     3     7     3    14     6     3     8     4     9     9     2    24     5     1
     0     0    30     0     0     0     2     3     6     2     3     0    39    15     0
     2     1     5     7    11     2     1     8     4     7    10     4    16    22     0
     2     0     3     3    20     6     3     6     1     7     3     4    22    19     1
     1     2    10     3     6     6     1     3     1     2     5     3    27    29     1
     1     2    10     2     3     1    33     1     1     6    15     0    11    14     0
     1     0    10     3     5     1     2    18     0     7     1     0    36    14     2
     4     2    13     1     6     5     7     7     5     8     4     2    25    10     1
     4     1     8     0     8     4     2    11     4    12    13     3    24     6     0
     1     2    10     0     6     1    12     2     1     7    43     0    10     5     0
     2     6     2     3    20     9     1     7     1     2     5    11    19    12     0
     0     0    16     0     2     2     3     6     2     5     1     0    45    17     1
     0     0    11     1     0     0     0     3     0     1     0     0    14    70     0
     2     2     5     3    11     5     4     8     3    10     8     6    27     5     1


accuracy =

   0.212667
