model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 7)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Select GPU (cuda) or CPU
model.to(device)

composed = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = Dataset(transform=composed, train=True)
validation_dataset = Dataset(transform=composed, train=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=15)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=10)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    [param for param in model.parameters() if param.requries_grad],
    lr=0.003,
)

for epoch in range(20):
    epoch_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        model.train()
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    correct = 0
    for x, y in validation_loader:
        x, y = x.to(device), y.to(device)
        model.eval()
        z = model(x)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y).sum().item()
    epoch_accuracy = correct / len(validation_dataset)
    