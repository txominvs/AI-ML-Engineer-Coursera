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

model = Custom_Regression(in_size=2, out_size=3)
x = torch.tensor([ # batch size = 3
    [0.1, 0.2],
    [1.3, 0.5],
    [1.1, 0.9],
])
z = model(x)
_, yhat = z.max(axis=1) # for each row, select the maximum column

criterion = nn.CrossEntropyLoss() # automatically apply SOFTMAX!
