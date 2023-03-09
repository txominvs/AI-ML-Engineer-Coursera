import torch
import torch.nn as nn # PyTorch Neural Network
from torch.utils.data import Dataset, DataLoader # Used to help create the dataset and perform mini-batch
import torch.nn.functional as F # Allows us to use activation functions

class Custom_dataset(Dataset):
    def __init__(self, K=3, N=500):
        x = np.zeros((how_many_samples, how_many_features))
        y = np.zeros(how_many_features, dtype='uint8')
        self.x = torch.from_numpy(x).type(torch.FloatTensor)
        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.len = how_many_features
    def __getitem__(self, index):    
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len

class Custom_network(nn.Module):
    def __init__(self, nodes_per_layer):
        super().__init__()
        self.hidden = nn.ModuleList()
        for current_layer_nodes, next_layer_nodes in zip(nodes_per_layer, nodes_per_layer[1:]):
            self.hidden.append(nn.Linear(current_layer_nodes, next_layer_nodes))
    def forward(self, x):
        for layer in self.hidden[ : -1]:  # apply activation to all BUT THE LAST
            x = F.relu(layer(x)) # F = stands for torch.nn.functional
        last_layer = self.hidden[-1]
        x = last_layer(x) # activation for the las layer is carried out by criterion=nn.CrossEntropyLoss()
        return x

train_loader = DataLoader(dataset=Custom_dataset(), batch_size=20)

model = Custom_network(nodes_per_layer=[
    2, # input size = how many features
    50, # neurons in the hidden layer
    3, # output size = how many classes
])
optimizer = torch.optim.SGD(model.parameters(),
                            lr=0.1,
                            momentum=0.2)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    for x, y in train_loader: # For batch in train laoder
        optimizer.zero_grad() # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
        yhat = model(x)
        loss = criterion(yhat, y)
        loss.backward() # Calculates the gradient value with respect to each weight and bias
        optimizer.step() # Updates the weight and bias according to calculated gradient value
    last_loss = loss.item()
    last_accuracy = (torch.max(model(validation_x), 1) [1] == validation_y).numpy().mean()
