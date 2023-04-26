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

# Convolution channels: matrix multiplication between kernels * input channels as rows
biases = conv.state_dict()['biases']
weights = conv.state_dict()['weight']
outputs = []
for out_channel in range(out_channels):
    temp = 0
    for in_channel in range(in_chanels):
        temp += convolution(image[in_channel], weights[out_channel][in_channel])
    temp += biases[out_channel]
    outputs.append(temp)

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

###
# Working example
###
class Data(torch.utils.data.Dataset):
    def __init__(self):
        self.x, self.y, self.len = ...
    def __getitem__(self,index):      
        return self.x[index],self.y[index]
    def __len__(self):
        return self.len

import torchvision.transforms as transforms
import torchvision.datasets as dsets
composed = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=composed)
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=composed)

# https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    kernel_size = kernel_size if type(kernel_size) is tuple else (kernel_size, kernel_size)
    h = floor((h_w[0] + 2*pad - dilation*(kernel_size[0]-1) - 1)/stride + 1)
    w = floor((h_w[1] + 2*pad - dilation*(kernel_size[1]-1) - 1)/stride + 1)
    return h, w

class Custom_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2) # shape = conv_output_shape((16,16), kernel_size=5, stride=1, pad=2, dilation=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # shape = conv_output_shape(shape, kernel_size=2, stride=2, pad=0, dilation=1)
        self.bn1 = nn.BatchNorm2d(num_features=16) # BATCH NORMALIZATION 2D num_features=out_channels

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2) # shape = conv_output_shape(shape, kernel_size=5, stride=1, pad=2, dilation=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # shape = conv_output_shape(shape, kernel_size=2, stride=2, pad=0, dilation=1)
        self.bn2 = nn.BatchNorm2d(num_features=32) # BATCH NORMALIZATION 2D num_features=out_channels

        self.fc1 = nn.Linear(32*4*4, 10) # shape = channels * 4*4
        self.bn3 = nn.BatchNorm1d(10)
    def forward(self,x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.bn1(x)

        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = self.bn2(x)

        x = x.view(x.size(0), -1) # Flatten 2D images and channels into VECTOR

        x = self.fc1(x)
        x = self.bn3(x)
        return x

model = Custom_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5_000)

for epoch in range(10):
    cost_per_epoch = 0
    for x, y in train_loader:
        model.train() # activate BATCH NORMALIZATION layers
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z,y)
        loss.backward()
        optimizer.step()
        cost_per_epoch += loss.item()
    
    correct = 0
    for x_test, y_test in validation_loader:
        model.eval() # stop changing BATCH NORMALIZATION parameters
        z = model(x_test)
        _, yhat = torch.max(z.data,1)
        correct += (yhat==y_test).sum().item()
    accuracy_per_epoch = correct/len(validation_dataset)

model.eval() # do not forget to freeze BATCH NORM layers in the end
plt.imshow(model.state_dict()['cnn1.weight'][out_index, in_index, :,:], cmap='seismic')
