# For downloading files use "wget -O" and for extracting files "unzip -xvzf"

import os.path

location_of_images = [
    os.path.join(folder_location, file_name)
    for file_name in os.listdir(folder_location)
    if file_name.endswith(".jpg")
].sort()


from PIL import Image
import matplotlib.pylab as plt
image_object = Image.open(location_of_images[0])
plt.imshow(image_object); plt.show()

##### KERAS part

negative_files = os.scandir('./Negative') # returns an iterator
file = next(negative_files)
file_name = file.name
image_data = plt.imread(file.path)

data_generator = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_input,
)
train_generator = data_generator.flow_from_directory(
    directory_path,
    batch_size=4,
    class_mode='categorical',
    seed=24,
    target_size=(image_resize, image_resize),
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=...,
    epochs=...,
    validation_data=validation_generator,
    validation_steps=...,
    verbose=1,
)