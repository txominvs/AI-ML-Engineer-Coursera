###
# Neural Networks
###

import torch
import torch.nn as nn

# Object oriented approach
class Custom_network(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, out_size)
    def forward(self, x):
        x = torch.sigmoid(self.linear_1(x))
        x = torch.sigmoid(self.linear_2(x))
        return x
model = Custom_network(in_size=1, hidden_size=2, out_size=1)
x = torch.tensor([
    [0],
])
yhat = model(x)
parameters = model.state_dict()

# Functional approach
model = nn.Sequential(
    nn.Linear(in_size, hidden_size),
    nn.Sigmoid(),
    nn.Linear(hidden_size, out_size),
    nn.Sigmoid(),
)
yhat = model(x)


def custom_BCE_loss(outputs, labels):
    return torch.mean(labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs))
criterion = custom_BCE_loss
# or equivalently
criterion = nn.BCELoss() # used for BINARY classification = class 0 or class 1
# For REGRESSIONS use the following:
criterion = nn.MSELoss()


optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# or
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


from torch.utils.data import Dataset, DataLoader
class Custom_dataset(Dataset):
    def __init__(self):
        self.x, self.y, self.len = ...
    def __getitem__(self,index):    
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len
train_loader = DataLoader(dataset=Custom_dataset(), batch_size=100)


# Train
for epoch in range(100):
    loss_per_epoch = 0
    for x, y in train_loader:
        yhat = model(x)
        loss = criterion(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted_label = yhat >= 0.5

        loss_per_epoch += loss.item()

# Overfitting = too many neurons, model complexity bigger than data complexity
# Underfitting = too few neurons, data complexity bigger than model complexity
def accuracy(model, data_set):
    return np.mean(data_set.y.view(-1).numpy() == (model(data_set.x)[:, 0] > 0.5).numpy())


###
# MULTI CLASS CLASSIFICATION
###
# binary classification loss = BCE loss
# multi class = Cross Entropy loss and **REMOVE SOFTMAX FROM LAST LAYER** https://stackoverflow.com/a/57521524/4965360
class Custom_network(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, out_size)
    def forward(self, x):
        x = torch.sigmoid(self.linear_1(x))
        x = self.linear_2(x)  # CrossEntropyLoss automatically applies SOFTMAX to last layer https://stackoverflow.com/a/57521524/4965360
        return x
model = Custom_network(in_size=28*28, hidden_size=100, out_size=10)
# or equivalently
model = torch.nn.Sequential(
    torch.nn.Linear(in_size, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, out_size), # CrossEntropyLoss automatically applies SOFTMAX to last layer https://stackoverflow.com/a/57521524/4965360
)


import torchvision.transforms as transforms
import torchvision.datasets as dsets
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)


criterion = nn.CrossEntropyLoss() # CrossEntropyLoss automatically applies SOFTMAX to last layer https://stackoverflow.com/a/57521524/4965360


optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for epoch in range(30):
    for x, y in train_loader: 
        z = model(x.view(-1, 28 * 28))
        loss = criterion(z, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_per_epoch = loss.data.item()

    correct = 0
    for x, y in validation_loader:
        z = model(x.view(-1, 28 * 28))
        _, label = torch.max(z, 1)
        correct += (label == y).sum().item()
    accuracy = correct / len(validation_dataset)


###
# Backpropagation
###
loss.backwards()
# loss = h(g(f(x))) so derivative_of_loss = derivative_of_h * derivative_of_g * derivative_of_f
# derivative_of_loss = derivative_of_last_layer * derivative_of_hidden * derivative_of_first layer

###
# Vanishing Gradient Problem
###
# Training the last layers of the network is easy, but training the first layers is hard
# Given that the derivative of the sigmoid activation function is less than 1, multiplying together
# many layer derivatives results in a very SMALL number which results is a very low learning rate
# for the first layers of the network. This is solved by using a ReLU activation function whose derivative is = 1


###
# Activation function showcase
###
yhat = torch.relu(z)
# or equivalently for model = nn.Sequential()
RELU = torch.nn.ReLU(); yhat = RELU(z)

# goes from 0 to +1 and shows the vanishing gradient problem
yhat = torch.sigmoid(z)
# or equivalently for model = nn.Sequential()
sig = torch.nn.Sigmoid(); yhat = sig(z)

# goes from (-1 to +1) and also shows the vanishing gradient problem
yhat = torch.tanh(z)
# or equivalently for model = nn.Sequential()
TANH = torch.nn.Tanh(); yhat = TANH(z)

# Object Oriented Approach uses: torch.activation(x)
class Custom_network(nn.Module):
    def __init__(self):
        ...
    def forward(self, x):
        x = torch.relu(x)
        ...
# or functional aproach uses: torch.nn.Activation()(x)
model = torch.nn.Sequential(
    torch.nn.ReLU(),
    ...,
)