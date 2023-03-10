import torch 
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision.datasets as dsets

###
### DATA AUGMENTATION
###
compose_rotate = transforms.Compose([
    transforms.Resize((16, 16)), # resizes the images
    transforms.RandomAffine(45), # randomly rotates it
    transforms.ToTensor() # converts it to a tensor
])

train_dataset_rotate = dsets.MNIST(root='./data', train=True,  download=True, transform=compose_rotate)
validation_dataset   = dsets.MNIST(root='./data', train=False, download=True, transform=compose_rotate)

###
### SAVE MODEL'S PARAMETERS
###
model = Custom_Neural_Network()

import os  
file_locaton = os.path.join(os.getcwd(), 'normal.pt')

checkpoint['model_state_dict'] = model.state_dict()
checkpoint['optimizer_state_dict'] = optimizer.state_dict()
torch.save(checkpoint, file_locaton)

checkpoint = torch.load(file_locaton)
model.load_state_dict(checkpoint['model_state_dict']) # Using the model parameters we saved we load them into a model to recreate the trained model
model.eval() # Setting the model to evaluation mode

for x, y in torch.utils.data.DataLoader(dataset=dataset, batch_size=1):
    z = model(x)
    _, yhat = torch.max(z, 1)
    if yhat != y:
        print("Missclassified!", x, y, yhat)