threshold_function = lambda x: 1 if x>0 else 0
sigmoid = lambda x: 1/(1+exp(-x))

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