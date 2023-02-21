# Image
- Height = number of rows and Width = number of cols
- Each pixel 0 ... 255
- 3 color channels: 0=red, 1=green, 2=blue

# Pillow (image manipulation in python)
```python
from PIL import Image

image_var = Image.open(file_name)
# image_var.format = "PNG"
# image_var.size = 512, 512
# image_var.mode = "RGB"

red_channel, green_channel, blue_channel = image_var.split()

image_var.show(title="Title of the image")
# or equivalently
import matplotlib.pyplot as plt
plt.imshow(image_var)

from PIL import ImageOps
grayscaled_image = ImageOps.grayscale(image) # image_var.mode = "L"
grayscaled_image.save(new_file_name_also_with_different_extension_possible)
grayscaled_image.quantize(2) # only two grayscale levels 0=no light 1=light

import numpy as np
image_as_array = np.array(image_var) # [red channel [rows x cols], green channel [rows x cols], red channel [rows x cols]]
```

# OpenCV (more difficult)
```python
import cv2
image_var = cv2.imread(image_location) # numpy array with shape (row=height col=width channels)
image_var = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE) # load grayscale

### WATCH OUT!
# PIL color channels 0=red 1=green 2=blue
# OpenCV color channel 0=blue 1=green 2=red

blue_channel, green_channel, red_channel = image_var[..., 0], image_var[..., 1], image_var[..., 2]

image_var = cv2.cvtColor(image_var, cv2.COLOR_BGR2RGB) # correct colorspace to RGB
image_var = cv2.cvtColor(image_var, cv2.COLOR_BGR2GRAY) # to grayscale
import matplotlib.pyplot as plt; plt.imshow(image_var)
cv2.imwrite(new_file_name, image_var)
```