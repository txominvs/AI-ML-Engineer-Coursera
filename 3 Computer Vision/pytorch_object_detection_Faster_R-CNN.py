model_ = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model_.eval()
for name, param in model_.named_parameters():
    param.requires_grad = False
# Save RAM
def model(x):
    with torch.no_grad():
        yhat = model_(x)
    return yhat

transformed_images_array = []
tensorizer_function = transforms.Compose([transforms.ToTensor()])
for image in images_array:
    image = Image.open(file_name_of_image)
    image = image.resize([int(0.5 * s) for s in image.size])
    tensor = tensorizer_function(image)
    transformed_images_array.push(tensor)

predictions_array = model(transformed_images_array)
for image, prediction in zip(transformed_images_array, predictions_array):
    for class_index, confidence, bounding_boxes in zip(prediction['labels'], prediction['scores'], prediction['boxes']):
        if confidence < threshold: continue
        human_readable_class = COCO_CLASS_NAMES[class_index]
        for bounding_box in prediction['boxes']: # for each detected object of a given class
            topleft_x, topleft_y, bottomright_x, bottomright_y = bounding_box

# Save RAM
del temp_variable
torch.cuda.empty_cache()
image.close(); del(image)

COCO_CLASS_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
] # 91 classes