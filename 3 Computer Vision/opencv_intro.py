import cv2

image = cv2.imread(file_name, flag=cv2.IMREAD_COLOR) # default optional flag, returns a NUMPY ARRAY with shape (row=height col=width channels)
load_image_into_grayscale = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE) # load grayscale

rows, columns, color_channel = image.shape #rows=height(top to bottom) cols=width(left to right) 0=blue 1=green 2=red

### WATCH OUT!
# PIL color channels are RGB 0=red 1=green 2=blue
# OpenCV color channel are BGR 0=blue 1=green 2=red

blue_channel, green_channel, red_channel = image[:,:, 0], image[:,:, 1], image[:,:, 2]
three_channels_in_a_single_image_vertically = cv2.vconcat([blue_channel, green_channel, red_channel])

cv2.imshow('image', image); cv2.waitKey(0); cv2.destroyAllWindows()
# or alternatively
color_corrected_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB); plt.imshow(color_corrected_image); plt.show()

cv2.imwrite(file_name_with_another_extension, image)

grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(grayscaled_image, cmap='gray'); plt.show()

reference_to_image_array = image # points to the same object in memory
copy_of_image = image.copy()