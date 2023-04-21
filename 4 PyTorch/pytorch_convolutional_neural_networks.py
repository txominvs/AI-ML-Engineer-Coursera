import torch
import torch.nn as nn

# CONVOLUTION: studies *relative* position of intensity values
# 1) Overlaying image and kernel and shift it: sum overlapping values
# 2) Apply broadcast-sum bias to each color channel

# TERMINOLOGY:
# zero padding = add rows and columns of zeros to the original image
# stride = step size
# activation map = result of convolution

# size of result = 1 + (rows_of_image - rows_of_kernel)/stride

conv = nn.Conv2d(
    in_channels=1, # how many colors does the input image have?
    out_channels=1,
    kernel_size=2,
    stride=2, # step size
)
layer_parameters = conv.state_dict()
image = torch.zeros(
    1, # batch size
    1, # color channels
    5, 5, # rows and cols
)
result = conv(image)

###
# MAX POOLING
###
# - only keep the largest value in a KxK region of the image
# - then shift to the left/bottom
max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
result = max_pooling(image)
# or equivalently
result = torch.max_pool2d(image, kernel_size=2, stride=2)

# WATCH OUT:
# - In max pooling: default stride = kernel_size
# - In convolution: default stride = 1

###
# Activation function
###
result = torch.relu(result)
# or equivalently
relu = nn.ReLU(); result = relu(result)