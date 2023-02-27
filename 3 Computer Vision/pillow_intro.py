# pip install Pillow
from PIL import Image

with Image.open(file_name) as image:
    # image.format = "PNG"
    # image.size = 512, 512
    # image.mode = "RGB"

    tv_noise_array = np.random.normal(0, 255, (rows, columns, 3)).astype(np.uint8)
    image = Image.fromarray(tv_noise_array) # creates a PIL Image from an array

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
    # WATCH OUT! For all arrays "dtype=uint8" so values overflow 256->0 257->1 258->2 and so on...
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

    # Affine transformations = A*X + B
    # Scaling + interpolation
    # Translation + fill zeros and larger
    # Rotation

    new_image = image.resize(size=(new_width, new_height))
    new_image = image.rotate(counter_clockwise_angle_in_degrees)

    # singular value decomposition for GrayScale images
    U, s, V = np.linalg.svd(grayscaled, full_matrices=True)
    S = np.diag(s)
    recover_image = U @ S @ V
    compressed_S = S[:, :n_components]
    compressed_V = V[:n_components, :]
    compresed_image = U @ compressed_S @ compressed_V

    from PIL import ImageFilter
    # low pass filter: smoother + nose removal + blur
    kernel = np.ones((5,5))/(5*5)
    kernel_filter = ImageFilter.Kernel((5,5), kernel.flatten())
    new_image = image.filter(kernel_filter)

    # gaussian blur
    new_image = image.filter(ImageFilter.GaussianBlur(radius=4))

    # sharpen details of image
    kernel = np.array([[-1,-1,-1], 
                    [-1, 9,-1],
                    [-1,-1,-1]]); kernel = ImageFilter.Kernel((3,3), kernel.flatten())
    new_image = image.filter(kernel)
    # sharpen edges
    new_image = image.filter(ImageFilter.SHARPEN)

    # edge detection
    new_image = image.filter(ImageFilter.EDGE_ENHANCE).filter(ImageFilter.FIND_EDGES)

    # median filter: noise removal without blurring edges!
    new_image = image.filter(ImageFilter.MedianFilter)