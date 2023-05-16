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