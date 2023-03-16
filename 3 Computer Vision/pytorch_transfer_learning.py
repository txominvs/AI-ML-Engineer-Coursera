import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader,random_split
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.nn as nn
torch.manual_seed(0)

# Compare the prediction and actual value:
def result(model,x,y):
    from PIL import Image
    with Image.open(imageName) as image:
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        transformed_image = transform(image)
        single_sample_batch = torch.unsqueeze(transformed_image, 0)
        z=model(single_sample_batch)
        _,yhat=torch.max(z.data, 1)
    return yhat.item() == y

# Define our device as the first visible cuda device if we have CUDA available:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("the device type is", device)

# Data augmentation and normalization
composed_augment = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # augmentation
    transforms.RandomRotation(degrees=5), # augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = dsets.MNIST(root='./data', train=True,  download=True, transform=composed_augment)
composed_normalize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
validation_dataset = dsets.MNIST(root='./data', train=False,  download=True, transform=composed_normalize)

# Pretrained model
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, n_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
validation_loader= torch.utils.data.DataLoader(dataset=val_set, batch_size=1)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=5, mode="triangular2")

accuracy_best=0
best_model_wts = copy.deepcopy(model.state_dict())
for epoch in range(n_epochs):
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        model.train() 

        z = model(x)
        loss = criterion(z, y)
        loss_value_for_the_current_minibatch = loss.data.item()
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

    scheduler.step()    
    learning_rate = optimizer.param_groups[0]['lr']

    for x_test, y_test in validation_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        model.eval()
        z = model(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()
    accuracy = correct / n_test

    if accuracy>accuracy_best:
        accuracy_best = accuracy
        best_model_wts = copy.deepcopy(model.state_dict())
    
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'model.pt')

# Load the model that performs best
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, n_classes)
model.load_state_dict(torch.load( "model.pt"))
model.eval()