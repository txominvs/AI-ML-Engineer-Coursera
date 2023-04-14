###
# SOFTMAX (soft argmax) REGRESSION
###

# Regression
# [row value for each class] = [row input features] * [matrix rows=classes cols=features] + [row bias for each class]

# Softmax (a, b, c)
# = NORMALIZE TO UNITARY VECTOR[ exp(a), exp(b), exp(c) ]
criterion = nn.CrossEntropyLoss()

# Argmax = which index of the vector has the largest value?

import torch
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision.datasets as dsets

class Custom_Regression(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
    def forward(self, x):
        return self.linear(x)
model = Custom_Regression(in_size=28*28, out_size=10)
print('W: ',list(model.parameters())[0].size())
print('b: ',list(model.parameters())[1].size())
# or equivalently
model = nn.Sequential(nn.Linear(1, 3))

x = torch.tensor([ # batch size = 3
    [0.1, 0.2],
    [1.3, 0.5],
    [1.1, 0.9],
])
z = model(x)
_, yhat = z.max(axis=1) # for each row, select the maximum column

####
# CRITERION = crossentropy: key of multi-class classification!
####
criterion = nn.CrossEntropyLoss() # automatically applies SOFTMAX! https://stackoverflow.com/a/57521524/4965360

from torch.utils.data import Dataset, DataLoader
class Data(Dataset):
    def __init__(self):
        self.x, self.y, self.len = ...
    def __getitem__(self, index):      
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len
train_loader = DataLoader(dataset = data_set, batch_size = 5)
# or equivalently
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)
plt.imshow(train_dataset[index_of_sample][0].numpy().reshape(28, 28), cmap='gray')


optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(300):
    for x, y in train_loader:
        optimizer.zero_grad()
        yhat = model(x.view(-1, 28 * 28))
        loss = criterion(yhat, y) # automatically applies SOFTMAX! https://stackoverflow.com/a/57521524/4965360
        loss.backward()
        optimizer.step()

        mini_batch_loss = loss.item()

        z =  model(x)
        _, yhat = z.max(1)
        mini_batch_accuracy = (y == yhat).sum().item()

        softmax_function = nn.Softmax(dim=-1)
        probabilities_for_each_class = softmax_function(z)

        confidence_of_guess, guessed_labels = torch.max(probabilities_for_each_class, dim=1).item()
