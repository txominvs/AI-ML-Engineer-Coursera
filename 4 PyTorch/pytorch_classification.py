threshold_function = lambda x: 1 if x>0 else 0
sigmoid = lambda x: 1/(1+exp(-x))
# bernulli_distribution = probability of sequence of events = P(a)^Na * P(b)^Nb
# Likelihood = P(model correct | having seen a sequence of events)
#            = P(sequence of events | using this model) = x^number_of_tails * (1-x)^number_of_heads
log_loss = log(likelihood) / all_tries = number_of_tails/all_tries*log(x) + number_of_heads/all_tries*log(1-x)
# This means log_loss=entropy since probability=exp(sequence_length*entropy)

###
# This might be the reason why we minimize LOG_LOSS in Classification instead of MEAN_SQUARED_ERROR
# We try to minimize LOG_LOSS = maximize LIKELIHOOD = maximize P(model is correct | having seen a sequence of training samples)
###

###
# CLASSIFICATION
###
z = lambda x: a*x + b 
# hyperplane: z(x) = 0
# one side of the hyperplane: z(x) >0
# the other side of the hyperplane: z(x) < 0

import torch.nn as nn
import torch

### Functional approach
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid(),
)

### Object oriented approach
class Logistic_regression(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
model = Logistic_regression(2)

### Use the models the same way
x = torch.tensor([ # batch size = 3
    [1.0, 1.0], # features = 2
    [3.0, 2.0],
    [2.0, 3.0],
])
yhat = model(x)
parameters = list(model.parameters()); parameters = model.state_dict()

criterion = nn.MSELoss() # = lambda yhat, y: torch.mean((yhat-y)**2)
criterion = nn.BCELoss() # = lambda yhat, y: -1 * torch.mean( y*torch.log(yhat) + (1-y)*torch.log(1-yhat) )

### Training example

from torch.utils.data import Dataset, DataLoader

class Custom_data(Dataset):
    def __init__(self):
        self.x, self.y, self.len = ...
    def __getitem__(self, index):      
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len

trainloader = DataLoader(dataset=Custom_data(), batch_size=3)
criterion = nn.BCELoss() # for regressions instead = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=2.0)

for epoch in range(epochs):
    for x, y in trainloader: 
        yhat = model(x)
        loss = criterion(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predicted_labels = yhat >= 0.5