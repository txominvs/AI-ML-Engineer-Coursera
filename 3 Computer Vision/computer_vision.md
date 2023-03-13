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

- z = b + SUM_i x_i*w_i
- probability = 1/(1+exp(-z)) sigmoid
- Loss function = difference between prediction and actual label. Not-smooth parabola with a minimum
    - Cross entropy loss = difference between sigmoid probability and actual label. Smooth parabola with a minimum
- Cost function = total loss = SUM of losses for all points in the dataset
- Gradient Descent: new value = value - step * slope
    - Low learning rate = too small or stuck
    - High learning rate = overshoot optimal value
- Batch gradient descent = loss function with all samples of the dataset at once
    - Iteration = weights have been updated
    - Batch size = how many samples before weights updated
    - Iterations = how many samples in dataset / batch size
    - Mini-batch gradient descent = loss function for a few datapoints at once
    - Epoch = all samples of the dataset have gone through training

# Multi-class: more than 2 classes
- Linea Classifier with Softmax: similar to `argmax_i(z_i)`,chooses the side of the hyperplane with the maximum value
    e^(value) / NORMALIZATION
- One-vs-rest (good for Support Vector Machines SVM)
- One-vs-one (good for Support Vector Machines SVM)
# Multi-label: checkboxes

# Feature extraction: HOG = Histogram of Oriented Gradients
1. Turn the image into GRAYSCALE
2. Divide image into patches (also called *cells*)
3. For each patch
    - For each pixel inside the patch:
        1. Compute the grayscale gradient at that point using Sobel
        2. Compute the angle where the gradient points to = atan(gradient_x, gradient_y)
    - Make a histogram of: how many gradients' angle is 0º-10º, how many between 10º-20º, ..., how many 350º-360º
4. To improve imbalance to highlights and shadows in the image, cells are block normalized
5. Feed data (features!) into Support Vector Machine

Other feature extraction methods: SURF and SIFT

# Image detection scheme
Feature extraction > SVM Nonlinear Kernel > Linear classifier

# Neural Network: free forward neural network = fully connected neural network
a bunch of "box functions" or "bump functions"
## Hiperplanes + logistic activation
1. Use 1 hyperplane to classify <-1 to 0 and >=-1 to 1
    activation function = 1/(1+e^(x-1))
2. Use another hyperplane to classify <1 to 1 and >=1 to 0
    activation function = 1/(1+e^(-1-x))
3. Apply weights +1 and -1
    activation function = threshold >=0.5
    this is another hyperplane classification

we have exactly recovered bump function = THRESHOLD[ 1/(1+e^(x-1)) - 1/(1+e^(-1-x)) ]
## Multi-class (ouput can be either cat, dog or duck): softmax
## Multi-label (non-mutually-exclusive labels like cat, pretty and red): binary sigmoid for each output
## Vanishing gradient problem:
Sigmoid activation function small gradient: backpropagating gradient results in vanishingly small numbers
ReLU activation function = max(0, x) solves this isssue by having a big gradient

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

# Convolutional Neural Networks (CNN)
Input > [Convolution > ReLU > Max Pooling >] Flatten [Linear Layer > ReLU] Linear Layer > Softmax
- Receptive Field: which region has affected the current pixel after the convolution?
- Max pooling: rescaling preserving features, invariant to small changes, increase receptive field

Image size after layer = floor( 1 + [size + 2*padding - dilation*(kernel size - 1) - 1] / stride )
- CNN layer: stride 1 padding 0 dilation 1
- Max pooling: stride=kernel size padding 0 dilation 1

## LeNet-5
Works for MINST dataset of handwritten digits
ImageNet classification dataset
- Grayscale image 32 x 32 x 1
- Repeat twice
    - Kernel 5x5 with stride=1
    - Max pooling 14 x 14
- 120 features for classificatioen
## AlexNet
- Kernel 11x11
## VGGNet
Very deep. Variants VGG16 layers and VGG19 layers
- 3x3 kernels twice, same receptive field less parameters
## ResNet
Vanishing gradient problem for deep networks. Deep residual learning has residual layers or skip layers perflow like layer:
layer output = layer input + weighted neuron

# Transfer Learning
Use previously trained CNNs aka. **pretrained models** as a feature generator.
Then change the last classification layers by our own softmax layers or even a Support Vector Machine.