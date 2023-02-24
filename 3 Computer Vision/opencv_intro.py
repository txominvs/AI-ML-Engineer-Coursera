import cv2

image = cv2.imread(file_name, flag=cv2.IMREAD_COLOR) # default optional flag, returns a NUMPY ARRAY with shape (row=height col=width channels)
# NUMPY ARRAY "dtype=uint8" so values overflow 256->0 257->1 258->2 and so on...
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

reference_to_image_array = image # points to the same object in memory id(old_var)==id(new_var)
copy_of_image = image.copy()

flipped_image = cv2.flip(image, flipcode) # flipcode = 0 upside down / 1 seen in a mirror / -1 horizontally and vertically flipped  
rotated_image = cv2.rotate(image, rotatecode) # rotatecode = cv2.ROTATE_90_CLOCKWISE cv2.ROTATE_90_COUNTERCLOCKWISE cv2.ROTATE_180

cropped_image = image[upper:lower+1, left:right+1, :]
cv2.rectangle(image, pt1=(left, upper), pt2=(right, lower), color=(0, 255, 0), thickness=3) 
cv2.putText(img=image, text='whatever you want to say', org=(left, lower), color=(255,0,0), fontFace=4, fontScale=5, thickness=2)

cv2.calcHist([image], channels=[ 0 ], mask=None, histSize=[256], ranges=[0,255])
# new color = alpha*color + beta WHERE alpha=contrast control AND beta=brightness control
# Also, the new color is clipped in the range [0, 255]
# Make image brighter alpha=3 beta=-200 # Invert image colors alpha=-1 beta=255
new_image = cv2.convertScaleAbs(image, alpha=-1, beta=255)
new_image = cv2.equalizeHist(image) # Histogram equalization: flatten histogram -> improved contrast
ret, new_image = cv2.threshold(image, threshold, maxval, cv2.THRESH_BINARY) # if color>=threshold: color = max else: color = 0
ret, new_image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU) # ret = automatically determined threshold

new_image = cv2.resize(image, None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC) # fx=horizontal scale factor / fy=vertical scale factor / cv2.INTER_CUBIC cv2.INTER_NEAREST
new_image = cv2.resize(image, (new_columns, new_rows), interpolation=cv2.INTER_CUBIC)
new_image = cv2.warpAffine(image, matrix, (new_columns:=output_width, new_rows:=output_height))
matrix = np.array([[1,0,horizontal_translation],[0,1,vertical_translation]])
matrix = cv2.getRotationMatrix2D(center=(width//2, height//2), angle=rotation_angle, scale=1)

# singular value decomposition for GrayScale images
U, s, V = np.linalg.svd(grayscaled, full_matrices=True)
S = np.diag(s)
recover_image = U @ S @ V
compressed_S = S[:, :n_components]
compressed_V = V[:n_components, :]
compresed_image = U @ compressed_S @ compressed_V