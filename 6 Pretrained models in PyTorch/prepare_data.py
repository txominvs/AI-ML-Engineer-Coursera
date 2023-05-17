import torch
from torch.utils.data import Dataset
from PIL import Image

class Custom_dataset(Dataset):
    def __init__(self, transform=None, train=True):
        import os.path
        self.location_of_images = [
            os.path.join(folder_location, file_name)
            for file_name in os.listdir(folder_location)
            if file_name.endswith(".jpg")
        ].sort()
        self.transform = transform
        self.y = torch.zeros([number_of_samples]).type(torch.LongTensor)        
        if train:
            self.y = self.y[0:30000]
            self.len = len(self.y)
        else:
            self.y = self.y[30000:]
            self.len = len(self.y)
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        image = Image.open(self.location_of_images[index])
        y = self.y[index]
        if self.transform:
            image = self.transform(image)
        return image, y

composed = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset_train = Custom_dataset(train=True, transform=composed)
dataset_validation = Custom_dataset(train=False, transform=composed)

train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=dataset_validation, batch_size=5000, shuffle=False)

import torch.nn as nn
class Custom_linear_module(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear_layer = nn.Linear(in_size, out_size)
    def forward(self, x):
        return self.linear_layer(x)

model = Custom_linear_module(in_size=227*227*3, out_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.1, lr=0.1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Select GPU (cuda) or CPU
model.to(device)

for epoch in range(5):
    loss_per_epoch = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device) # Send data to GPU
        z = model(batch_x.view(-1, 227*227*3))
        loss = criterion(z, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_per_epoch += loss.item()

    correct = 0
    for x, y in validation_loader:
        x, y = x.to(device), y.to(device) # Send data to GPU
        z = model(x.view(-1, 227*227*3))
        _, label = torch.max(z, 1)
        correct += (label == y).sum().item()
    accuracy_per_epoch = 100 * (correct / len(dataset_validation))
    print(f"{accuracy_per_epoch = }")

#### KERAS

import keras
from keras.preprocessing.image import ImageDataGenerator
# Inside "dataset_dir" images should be organized
# into folders depending on their category. So
# we would have "dataset_dir/class_1"
# and "dataset_dir/class_2" and so on...
data_generator = ImageDataGenerator(
    rescale=1./255,
)
image_generator = data_generator.flow_from_directory(
    dataset_dir,
    batch_size=4,
    class_mode='categorical',
    seed=24,
)
first_batch_images, first_batch_labels = image_generator.next()
# or equivalently
first_batch_images, first_batch_labels = next(image_generator)
import matplotlib.pyplot as plt
for image in first_batch_images:
    plt.imshow(image)
plt.show()
# or equivalently
for batch_images, batch_labels in image_generator:
    for image, label in zip(batch_images, batch_labels):
        ...

## ResNet model in Keras

import keras
data_generator = ImageDataGenerator(
    preprocessing_function=keras.applications.resnet50.preprocess_input,
)
train_generator = data_generator.flow_from_directory(
    'concrete_data_week3/train',
    target_size=(224, 224),
    batch_size=100,
    class_mode='categorical',
)
validation_generator = data_generator.flow_from_directory(
    'concrete_data_week3/valid',
    target_size=(224, 224),
    batch_size=100,
    class_mode='categorical',
)
model = keras.models.Sequential([
    keras.applications.ResNet50(
        include_top=False,
        pooling='avg',
        weights='imagenet',
    ),
    keras.layers.Dense(2, activation='softmax'),
])
model.layers[0].trainable = False
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
fit_history = model.fit_generator(
    train_generator,
    epochs=2,
    validation_data=validation_generator,
    verbose=1,
)
model.save('classifier_resnet_model.h5')
