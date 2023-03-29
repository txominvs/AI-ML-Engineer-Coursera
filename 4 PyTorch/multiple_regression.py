# Multiple linear regression:
# y = b + x_i w_i
#   = matrix[x_i inputs as rows] * column_matrix[w_i] + column_matrix[b b b b]

import torch
from torch import nn

x = torch.tensor([
    [1.0, 1.0], # first sample
    [3.0, 2.0], # second sample
    [4.0, 1.0], # third sample
])

### Functional approach
model = nn.Linear(in_features=2, out_features=1)
yhat = model(x)

### Hand-made approach
w = torch.tensor([[2.0], [3.0]], requires_grad=True)
b = torch.tensor([[1.0]], requires_grad=True)
def forward(x):
    yhat = torch.mm(x, w) + b
    return yhat
yhat = forward(x)

### Object oriented approach
class linear_regression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        return self.linear(x)
yhat = model(x)
parameters = list(model.parameters()); parameters = model.state_dict()