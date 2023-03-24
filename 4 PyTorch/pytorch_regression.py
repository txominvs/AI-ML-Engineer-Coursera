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
# Iteration = when are weights updated
# Epoch = all samples have gone through training

import torch

#
# Hand-made model
#
x = torch.linspace(0, 2, steps=10).view(-1, 1) # [[0], [1], [2], ...]
y = -3*x + 5 + 0.1*torch.randn( x.size() ) # w=-3 and b=5
w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(-1.0, requires_grad = True)
def forward(x): # evaluate model
    return b + w*x
def criterion(yhat, y): # calculate loss
    return torch.mean((yhat-y)**2)
learning_rate = 0.2
loss_per_epoch = []
for epoch in range(50):
    yhat = forward(x)
    loss = criterion(yhat, y)
    loss.backward()
    w.data = w.data - learning_rate*w.grad.data
    w.grad.data.zero_()
    b.data = b.data - learning_rate*b.grad.data
    b.grad.data.zero_()
    loss_per_epoch.append( loss.item() )
print(w, b)
print(loss_per_epoch)

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