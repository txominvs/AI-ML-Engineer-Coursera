# pip install Pillow
from PIL import Image

with Image.open(file_name) as image:
    # image.format = "PNG"
    # image.size = 512, 512
    # image.mode = "RGB"


    image.show(title="Title of the image") # open default viewer
    import matplotlib.pyplot as plt; plt.imshow(image); plt.show() # BEST! show image object
    image_array = np.array(image); plt.imshow(image_array); plt.show()

    width, height = image.size # width=cols height=rows

    access_to_pixel_color = image.load()
    color_channels = access_to_pixel_color[pixel_y, pixel_x]

    red_channel, green_channel, blue_channel = image.split()

    image.save(new_name_with_different_extension)

    from PIL import ImageOps
    grayscaled = ImageOps.grayscale(image)
    # grayscaled_image.mode == "L"

    down_sampled = grayscaled.quantize(256 // 2) # instead of 0...255 turn it into 0...127

    copy_of_image = image.copy()
    
    import numpy as np # IMAGE AS NUMPY ARRAY [red channel [rows x cols], green channel [rows x cols], red channel [rows x cols]]
    reference_access_to_pixel_values =  np.asarray(image) # points to same location in memory
    another_reference =                 reference_access_to_pixel_values # id(old_variable) == id(new_variable)
    copy_of_image_into_array =          np.array(image)
    yet_another_copy =                  reference_access_to_pixel_values.copy()

    rows, columns, color_channel = copy_of_image_into_array.shape #rows=height(top to bottom) cols=width(left to right) 0=red 1=green 2=blue

    plt.imshow(copy_of_image_into_array); plt.show() # automatically show in RGB
    plt.imshow(copy_of_image_into_array[:,:,0], cmap='gray') # just red channel (grayscaled)

    vertically_flipped = ImageOps.flip(image)
    vertically_flipped = image.transpose(Image.FLIP_TOP_BOTTOM)
    # Image.FLIP_LEFT_RIGHT,
    # Image.FLIP_TOP_BOTTOM,
    # Image.ROTATE_90,
    # Image.ROTATE_180,
    # Image.ROTATE_270,
    # Image.TRANSPOSE, 
    # Image.TRANSVERSE
    mirror_leftright = ImageOps.mirror(image)
    cropped_image = np.array(image)[upper:lower+1, left:right+1, :]
    cropped_image = image.crop((left, upper, right, lower))

    from PIL import ImageDraw
    image_canvas = ImageDraw.Draw(im=image)
    image_canvas.rectangle(xy=[left, upper, right, lower],fill="red")
    from PIL import ImageFont
    image_canvas.text(xy=(left, bottom),text="anything you want to write",fill=(0,0,0))
    
    image.paste(small_image, box=(left, upper))
    image_array[upper:lower, left:right, :] = small_image_array[upper:lower, left:right, :]