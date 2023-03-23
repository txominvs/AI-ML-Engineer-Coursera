# Linear regression: target = bias + weight * predictor
# Assumption: additive Gaussian noise with MEAN=0 STD=SIGMA
# Loss(parameter space) = (datapoint - prediction)^2
# Cost = MEAN SQUARE ERROR = AVERAGE[ Loss ] for all datapoints

# Newton method (finds zero):       x -> x - f(x) / (df/dx)
# Gradient descent  = find minimum = find zero derivative
#                   = apply newthon method to d[loss]/dw
#                   = w - loss'(w) / loss''(w) where GAMMA = 1/loss''(w) = LEARNING RATE
# Learning rate too LARGE = overshoot, too SMALL = local minima
# Batch = all samples in the training set
# Batch gradient descent: calculate loss with all samples -> then, update weights

import torch

#
# Hand-made model
#
w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(-1.0, requires_grad = True)
def forward(x):
    return b + w*x
x = torch.tensor([[1.0], [2.0]]) # batch_size = 2
yhat = forward(x)

#
# Functional approach
#
from torch.nn import Linear
torch.manual_seed(1)
linear_layer = Linear(in_features=1, out_features=1, bias=True)
parameters = list(linear_layer.parameters()); parameters = linear_layer.state_dict(); parameters = (linear_layer.weight, linear_layer.bias)
x = torch.tensor([[1.0], [2.0]]) # batch_size = 2
yhat = linear_layer(x)

#
# Object-oriented approach
#
from torch import nn
class Custom_Linear_Regression(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        out = self.linear(x)
        return out
linear_model = Custom_Linear_Regression(in_features=1, out_features=1)
parameters = list(linear_model.parameters()); parameters = linear_model.state_dict()
x = torch.tensor([[1.0], [2.0]]) # batch_size = 2
yhat = linear_model(x)