import torch # PyTorch Library
import torch.nn as nn # PyTorch Neural Network
from torch.utils.data import Dataset, DataLoader # Used to help create the dataset and perform mini-batch

class Net(nn.Module):
    def __init__(self, input_size, neurons_in_hidden_layer, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, neurons_in_hidden_layer)
        self.linear2 = nn.Linear(neurons_in_hidden_layer, output_size)
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x)) 
        x = torch.sigmoid(self.linear2(x))
        return x

class Custom_data(Dataset):
    def __init__(self):
        # Feature and target arrays
        self.x = torch.zeros((how_many_samples, 2))
        self.y = torch.zeros((how_many_samples, 1))
        # Populate features
        self.x[sample_index, :] = torch.Tensor([0.0, 1.0])
        self.y[sample_index, 0] = torch.Tensor([1.0])
        # Add noise if needed
        self.x = self.x + 0.01 * torch.randn((N_s, 2)) # add noise
        self.len = how_many_samples
    def __getitem__(self, index):    
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len

criterion = nn.BCELoss() # Binary Cross Entropy
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # gradient descent with learning rate = 0.1
train_loader = DataLoader(dataset=Custom_data(), batch_size=1) # mini-batch size = 1

for epoch in range(500):
    epoch_loss = 0 # Total loss over epoch
    for x, y in train_loader: # For batch in train laoder
        optimizer.zero_grad() # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
        yhat = model(x)
        loss = criterion(yhat, y) # Measures the loss between prediction and acutal Y value
        loss.backward() # Calculates the gradient value with respect to each weight and bias
        optimizer.step() # Updates the weight and bias according to calculated gradient value
        epoch_loss += loss.item() # Cumulates loss 
    epoch_accuracy = np.mean( data_set.y == (model(data_set.x)[:, 0] > 0.5).numpy() )