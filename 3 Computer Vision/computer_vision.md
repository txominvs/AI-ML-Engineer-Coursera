# Image
- Height = number of rows and Width = number of cols
- Each pixel 0 ... 255
- 3 color channels: 0=red, 1=green, 2=blue

# Pillow (image manipulation in python)
# OpenCV (more difficult)
> WATCH OUT!
> PIL color channels are RGB 0=red 1=green 2=blue
> OpenCV color channel are BGR 0=blue 1=green 2=red

# Image classification

# Your IBM Cloud Feature Code:



Copy the feature code below or click Activate Your Trial to get started with IBM Cloud.

7393dd6b95ad645fe65de1deac125fdc

Account ID: af38a434a80f4e0792cd506dd54cd5d6

# k-Nearest Neighbors (KNN)

1. distance(target and datapoints) = sqrt( (target feature - datapoints feature)**2 + ... )
2. Choose the "k" datapoints in the training set with the lowest distance
3. Class of target = the most frequent class among those "k" datapoints, aka **majority vote**

accuracy = correct guesses / total guesses
slow and too simple

- Training set: used to calculate distances(target and datapoint in training set)
- Validation set: used to choose the optimal value for "K"
- Test set: how well will it perform with data it has never seen?

# Feature Extraction
# Linear Classifier
