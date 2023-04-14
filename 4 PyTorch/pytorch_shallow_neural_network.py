###
# Neural Networks
###

import torch
import torch.nn as nn

# Object oriented approach
class Custom_newtork(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, out_size)
    def forward(self, x):
        x = torch.sigmoid(self.linear_1(x))
        x = torch.sigmoid(self.linear_2(x))
        return x
model = Custom_newtork(in_size=1, hidden_size=2, out_size=1)
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
# For REGRESSIONS use the following: criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Train
cost = []
for epoch in range(100):
    loss_per_epoch = 0
    for x, y in zip(X, Y):
        yhat = model(x)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predicted_label = yhat >= 0.5

        loss_per_epoch += loss.item() 