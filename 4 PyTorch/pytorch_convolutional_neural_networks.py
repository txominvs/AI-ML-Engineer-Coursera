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

###
# Running on a GPU
###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Select GPU (cuda) or CPU

tensor_on_GPU = torch.tensor([1, 2, 3, 4]).to(device) # Send a tensor to GPU

model = Custom_CNN()
model.to(device) # Convert model's layers to GPU tensors

for epoch in range(100):
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device) # Send data to GPU
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backwards()
        optimizer.step()

###
# Pre-trained models with TORCH-VISION
###
# ResNet18 residual network with skip connections
import torchvision.models as models
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 7) # add fully connected layer with requires_grad=True by default

from torchvision import transforms
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
composed = transforms.Compose([
    transforms.Resize(244),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_dataset = Custom_dataset(transform=composed, train=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=15)
validation_dataset = Custom_dataset(transform=composed)
validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=20)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    [param for param in model.parameters() if param.requires_grad],
    lr = 0.003
)

for epoch in range(20):
    for x, y in train_loader:
        model.train() # Training mode
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, x)
        loss.backward()
        optimizer.step()

        loss_per_iteration = loss.item()
    
    correct = 0
    for x, y in validation_loader:
        model.eval() # Freeze layers
        z = model(x)
        _, yhat = torch.max(z, 1)
        correct += (yhat == y).sum().item()
    accuracy_per_epoch = correct / len(validation_dataset)


### PyTorch standard preprocessing https://pytorch.org/hub/pytorch_vision_resnet/
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), # crop data from [0 255] to [0. 1.]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
single_sample_minibatch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
