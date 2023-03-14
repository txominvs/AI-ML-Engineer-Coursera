import cv2

pretrained_model_file_name = 'https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml'
detector = cv2.CascadeClassifier(pretrained_model_file_name)

example_image_file_name = "https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/CV0101/Dataset/car-road-behind.jpg"
image = cv2.imread(image_name)

object_list = detector.detectMultiScale(image)
for obj in object_list: # for each detected car, draw a rectangle around it 
    (x, y, w, h) = obj
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (255, 0, 0), # Rectangle border (blue green red color scheme) = BLUE
        2, # line thickness
    )