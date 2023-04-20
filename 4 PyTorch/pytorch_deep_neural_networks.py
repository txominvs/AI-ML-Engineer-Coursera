import torch 
import torch.nn as nn

class Network_with_relus(nn.Module):
    def __init__(self, input_size, hiddens_1, hiddens_2, output_size):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hiddens_1)
        self.linear_2 = nn.Linear(hiddens_1, hiddens_2)
        self.linear_3 = nn.Linear(hiddens_2, output_size)
    def forward(self, x):
        x = torch.relu(self.linear_1(x)) # torch.relu, torch.tanh, torch.sigmoid
        x = torch.relu(self.linear_2(x))
        x = self.linear_3(x)
        return x
# or equivalently
class Network_with_relus(nn.Module):
    def __init__(self, *list_of_neurons):
        super().__init__()
        # In order for "model.parameters()" to see the other layers, they must
        # be saved inside a nn.ModuleList() and not an ordinary list()
        # https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/4
        self.layers = nn.ModuleList()

        for layer_in, layer_out in zip(list_of_neurons, list_of_neurons[1:]):
            self.layers.append(nn.Linear(layer_in, layer_out))
    def forward(self, x):
        hidden_layers, last_layer = self.layers[:-1], self.layers[-1]
        
        for layer in hidden_layers:
            x = torch.relu(layer(x))
        
        # no activation function for the last layer
        # because we use criterion = nn.CrossEntropyLoss()
        x = last_layer(x)
        return x
model = Network_with_relus(28*28, 50, 50, 10)

import torchvision.transforms as transforms
import torchvision.datasets as dsets
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for i, (x, y) in enumerate(train_loader):
        z = model(x.view(-1, 28 * 28))
        loss = criterion(z, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_per_epoch = loss.item()

    correct = 0
    for x, y in validation_loader:
        z = model(x.view(-1, 28 * 28))
        _, label = torch.max(z, 1)
        correct += (label == y).sum().item()
    accuracy_per_epoch = 100 * (correct / len(validation_dataset))

###
# Dropout
###
# After the activation function, randomly set to zero the output of each neuron of a layer
# When the model is very complex --> overfitting is likely
# Dropout prevents overfitting by shutting of neurons in a layer (randomly) --> reducing model complexity
# P = probability of shutting down a neuron
# 1/(1-P) = given that on average only 1-P neurons are active
#           the magnitude of the output of a layer is smaller
#           compared to not having DROPOUT. That is why the 
#           output has to be rescaled to be bigger by this factor


# Create Net Class

class Net(nn.Module):
    def __init__(self, *layer_sizes, dropout_probability):
        super(Net, self).__init__()
        self.drop = nn.Dropout(p=dropout_probability)
        self.linear_1, self.linear_2, self.linear_3 = ...
    def forward(self, x):
        x = torch.relu(self.linear_1(x))
        x = self.drop(x)
        x = torch.relu(self.linear_2(x))
        x = self.drop(x)
        x = self.linear_3(x) # no dropout nor activation for in the last layer
        return x
# or equivalently
model = nn.Sequential(
    nn.Linear(...),
    nn.ReLU(),
    nn.Dropout(dropout_probability),
    nn.Linear(...),
    nn.ReLU(),
    nn.Dropout(dropout_probability),
    nn.Linear(...), # no dropout nor activation for the last layer
)

model.train() # activate DROPOUT layers

optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # better than optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss() # or criterion = torch.nn.MSELoss()

for epoch in range(500):
    yhat = model(x)
    loss = criterion(yhat, y)
    training_loss = loss.item()

    model.eval() # no DROPOUT will ocur
    validation_loss = criterion(model(val_x), val_y).item()

    model.train() # activate DROPOUT layers
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval() # no DROPOUT will ocur
_, yhat = torch.max(model(test_x), 1)
test_accuracy = (yhat == test_y).sum() / len(test_x)

###
# How to initialize weights
###

# If all the neurons had the same parameters, they would all give the same
# output and such, they would all have the same gradient value. Thus the
# training would be identical for all of them and will end up having the
# same values after the training --> poor model

# Also, if they are intialized with too small values --> not enough variation
# If the initialization too large --> activation function vanishes --> vanishing gradient problem
# A lot of neurons in previous layer --> value too large in current neuron --> vanishing gradient

linear = nn.Linear(input_size, output_size)
# Default initialization
# weights = uniform_random(low=-1/sqrt(neurons_in_layer), high=+1/sqrt(neurons_in_layer))
stdv = 1. / math.sqrt(linear.weight.size(1))
linear.weight.data.uniform_(-stdv, stdv)
linear.self.bias.data.uniform_(-stdv, stdv)
# Manual intialization
model.state_dict()["linear_layer_name.weight"][0] = .23
model.state_dict()["linear_layer_name.bias"][0] = -0.44
# Xavier initialization for TANH activation
# weights = uniform_random(low=-sqrt(6)/sqrt(neurons_in_layer + neurons_next_layer), high=+1/sqrt(neurons_in_layer + neurons_next_layer))
torch.nn.init.xavier_uniform_(linear.weight)
torch.nn.init.zeros_(linear.bias)
# He initialization for ReLU activation
torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")

###
# MOMENTUM TERM in gradient descent
###

# new velocity = d[loss]/d[weights] + momentum * old velocity
# new weights = old weights - learning rate *  new velocity

# Momentum helps avoid: saddle points, local minima and vanishing gradients
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5) # momentum term

###
# Example: FIND THE MINIMUM OF A POLYNOMIAL USING **PYTORCH**
###
class Fourth_order_polynomial(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False) # the weight will act as the X value to be optimized
    def forward(self):
        x = self.linear(torch.tensor([[1.0]]))
        yhat = 2*(x**4) -9*(x**3) -21*(x**2) + 88*x + 48
        return yhat
polynomial_model = Fourth_order_polynomial()
# Draw the polynomial
x_values = torch.arange(-4., 6., 0.1)
y_values = []
for polynomial_model.state_dict()['linear.weight'][0] in x_values:
    y_values.append( polynomial_model().item() )
plt.plot(x_values.numpy(), y_values)
# Find the global minimum & add noise for difficulty
optimizer = torch.optim.SGD(polynomial_model.parameters(), lr=0.001, momentum=0.9) # momentum term
polynomial_model.state_dict()['linear.weight'][0] = 6.0 # starting point
optimized_x_values, optimized_y_values = [], []
for n in range(100):
    loss = polynomial_model()
    x_value = polynomial_model.state_dict()['linear.weight'][0].detach().data.item()
    y_value = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimized_x_values.append(x_value)
    optimized_y_values.append(y_value)
plt.scatter(optimized_x_values, optimized_y_values)
