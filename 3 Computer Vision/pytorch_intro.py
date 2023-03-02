import torch # PyTorch Library
import torch.nn as nn # PyTorch Neural Network
from torch.utils.data import Dataset, DataLoader # Used to help create the dataset and perform mini-batch

class Data(Dataset): # create some random dataset
    def __init__(self):
        self.x = torch.arange(-1, 1, 0.1).view(-1, 1) # create a single feature for each sample: [[0], [0.2], [-0.1], ...]
        self.y = torch.zeros(self.x.shape[0], 1); self.y[self.x[:, 0] > 0.2] = 1 # create target labels for each sample: [["cat"], ["dog"], ...]
        self.len = self.x.shape[0]
    def __len__(self): # compatible with len(Data())
        return self.len
    def __getitem__(self, index): # data can be accessed like this: sample_features, sample_label = Data()[131]
        return self.x[index], self.y[index]
data_set = Data()

class logistic_regression(nn.Module): # base class for all neural networks
    def __init__(self, n_inputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, 1) # Single layer of Logistic Regression with number of inputs being n_inputs and there being 1 output
    def forward(self, x):
        yhat = torch.sigmoid(self.linear(x)) # activation function sigmoid
        return yhat
model = logistic_regression(n_inputs=1)

x = torch.tensor([-1.0]); sigma = model(x) # Feed custom data to the NN
criterion = nn.BCELoss(); loss = criterion(sigma, actual_label) # compute error
yhat = model(data_set.x); yhat = torch.round(yhat) # making predictions

trainloader = DataLoader(dataset=data_set, batch_size=10) # mini-batch size
dataset_iter = iter(trainloader); X,y=next(dataset_iter) # batch_size=10 samples will fed

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

### WORKING SAMPLE
model = logistic_regression(1)
criterion = nn.BCELoss()
trainloader = DataLoader(dataset = data_set, batch_size = 5)
optimizer = torch.optim.SGD(model.parameters(), lr = .01)
loss_values = []
for epoch in range(500): # epochs=500
    for x, y in trainloader: # each batch in the training data
        yhat = model(x)
        loss = criterion(yhat, y)
        optimizer.zero_grad() # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
        loss.backward() # Calculates the gradient value with respect to each weight and bias
        optimizer.step() # Updates the weight and bias according to calculated gradient value
        W = list(model.parameters())[0].item(); B = list(model.parameters())[1].item(); LOSS = loss.tolist()
w = model.state_dict()['linear.weight'].data[0]; b = model.state_dict()['linear.bias'].data[0]
