import torch 
import torch.nn as nn # PyTorch Neural Network

class Network_with_relus(nn.Module):
    def __init__(self, input_size, neurons_first_hidden_layer, neurons_second_hidden_layer, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, neurons_first_hidden_layer)
        self.linear2 = nn.Linear(neurons_first_hidden_layer, neurons_second_hidden_layer)
        self.linear3 = nn.Linear(neurons_second_hidden_layer, output_size)
    def forward(self, x):
        x = torch.relu(self.linear1(x)) # avoids vanishing gradient problem that ocurs with torch.sigmoid(self.linear1(x))
        x = torch.relu(self.linear2(x)) # avoids vanishing gradient problem that ocurs with torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x

train_dataset       = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=torchvision.transforms.ToTensor())
validation_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True) # Batch size is 2000 and shuffle=True means the data will be shuffled at every epoch
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False) # Batch size is 5000 and the data will not be shuffled at every epoch

criterion = nn.CrossEntropyLoss()
model = Network_with_relus(28*28, 50, 50, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    for x, y in train_loader: # For each batch in the train loader
        optimizer.zero_grad() # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
        flattened_inputs = x.view(-1, 28 * 28)
        z = model(flattened_inputs)
        loss = criterion(z, y)
        loss.backward() # Calculates the gradient value with respect to each weight and bias
        optimizer.step() # Updates the weightes and biases according to calculated gradient value
        iteration_training_loss = loss.data.item()
    
    # Counter to keep track of correct predictions
    correct = 0
    for x, y in validation_loader:
        flattened_inputs = x.view(-1, 28 * 28)
        z = model(flattened_inputs)
        _, label = torch.max(z, 1)
        correct += (label == y).sum().item()
    epoch_accuracy = 100 * (correct / len(validation_dataset))