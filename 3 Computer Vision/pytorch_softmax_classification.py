import torch 
import torch.nn as nn # PyTorch Neural Network
import torchvision.transforms as transforms # Allows us to transform data
import torchvision.datasets as dsets # Allows us to get the digit dataset

train_dataset       = dsets.MNIST(root='./save_them_here', train=True,  download=True, transform=transforms.ToTensor())
validation_dataset  = dsets.MNIST(root='./save_them_here', train=False, download=True, transform=transforms.ToTensor())
plt.imshow(train_dataset[index_of_the_datapoint][0].numpy().reshape(28, 28), cmap='gray'); plt.show()
train_dataset[0][0].view(-1,28*28).shape

class linear_model_trained_for_softmax(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        z = self.linear(x)
        return z

model = linear_model_trained_for_softmax(input_dim, output_dim)
w = list(model.parameters())[0]; w=model.state_dict()['linear.weight'].data; b = list(model.parameters())[1]

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss() # This is where the SoftMax occurs, it is built into the Criterion Cross Entropy Loss
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5_000)

X = train_dataset[index_of_sample:index_of_sample+1][0].view(-1, 28*28)
prediction = model(X); label = train_dataset[index_of_sample:index_of_sample+1][1]
softmax = nn.Softmax(dim=-1); confidence_in_prediction = torch.max(softmax(prediction)).item()

logloss = criterion(model_output, actual)
softmax = nn.Softmax(dim=1); probability = softmax(model_output); logloss = -1*torch.log(probability[0][label])

#####
# WORKED OUT EXAMPLE
#####

n_epochs = 10
loss_list = []; accuracy_list = []

for epoch in range(n_epochs):
    for x, y in train_loader: # For each batch in the train loader
        optimizer.zero_grad() # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
        linear_prediction = model(x.view(-1, 28 * 28))
        loss = criterion(linear_prediction, y)
        loss.backward() # Calculates the gradient value with respect to each weight and bias
        optimizer.step() # Updates the weight and bias according to calculated gradient value
    loss_list.append(loss.data)

    correct = 0
    for x_test, y_test in validation_loader: # For each batch in the validation loader
        linear_prediction = model(x_test.view(-1, 28 * 28))
        _, argmax_result = torch.max(linear_prediction.data, 1)
        correct += (argmax_result == y_test).sum().item()
    accuracy = correct / len(validation_dataset)
    accuracy_list.append(accuracy)
