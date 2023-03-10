import torch
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision.datasets as dsets

transformation_operations = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.ToTensor()
    ])
train_dataset       = dsets.MNIST(root='./data', train=True,  download=True, transform=transformation_operations)
validation_dataset  = dsets.MNIST(root='./data', train=False, download=True, transform=transformation_operations)

class Custom_CNN(nn.Module):
    def __init__(self, first_layer_kernels, second_layer_kernels):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=first_layer_kernels, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=first_layer_kernels, out_channels=second_layer_kernels, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(second_layer_kernels * 4 * 4, 10)

    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1) # Flattening layer
        x = self.fc1(x)
        return x

model.state_dict()['cnn1.weight'] # kernels of the FIRST convolutional layer: [kernel index, channel index, kernel rows, kernel cols]
model.state_dict()['cnn2.weight']

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    total_loss_epoch = 0
    for x, y in train_loader: # For each batch in train loader
        optimizer.zero_grad() # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
        z = model(x)
        loss = criterion(z, y)
        loss.backward() # Calculates the gradient value with respect to each weight and bias
        optimizer.step() # Updates the weight and bias according to calculated gradient value
        total_loss_epoch += loss.data
    
    correct=0
    for x_test, y_test in validation_loader: # For each batch in validation loader 
        z = model(x_test)
        _, yhat = torch.max(z.data, 1) # The class with the max value is the one we are predicting
        correct += (yhat == y_test).sum().item() # In the current batch, how many predictions equal the actual label?
    accuracy = correct / len(validation_dataset)
     
# Find the misclassified samples
for x, y in torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1):
    z = model(x)
    _, yhat = torch.max(z, 1)
    if yhat != y:
        print("yhat: ",yhat)