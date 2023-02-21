# pip install Pillow
from PIL import Image

with Image.open(file_name) as image:


    image.show() # open default viewer
    import matplotlib.pyplot as plt; plt.imshow(image); plt.show() # BEST!

    width, height = image.size # width=cols height=rows


    assert image.mode == "RGB"
    access_to_pixel_color = image.load()
    color_channels = access_to_pixel_color[pixel_y, pixel_x]

    red_channel, green_channel, blue_channel = image.split()

    image.save(new_name_with_different_extension)


    from PIL import ImageOps
    grayscaled = ImageOps.grayscale(image)
    assert grayscaled.mode == "L"

    down_sampled = grayscaled.quantize(256 // 2) # instead of 0...255 turn it into 0...127

    
    import numpy as np
    reference_access_to_pixel_values =  np.asarray(image) # points to same location in memory
    copy_of_image_into_array =          np.array(image)
    yet_another_copy =                  reference_access_to_pixel_values.copy()

    rows, columns, color_channel = copy_of_image_into_array.shape #rows=height(top to bottom) cols=width(left to right) 0=red 1=green 2=blue

    plt.imshow(copy_of_image_into_array); plt.show() # automatically show in RGB
    plt.imshow(copy_of_image_into_array[:,:,0], cmap='gray') # just red channel (grayscaled)