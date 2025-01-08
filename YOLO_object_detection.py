import cv2
import numpy as np
import os

# Paths to the files
cfg_file = r"C:\Users\Administrator\Downloads\Compressed\yolo_objectDetection_imagesCPU-master\yolo_objectDetection_imagesCPU-master\yolov3.cfg"  # Path to the YOLOv3 configuration file
weights_file = r"C:\Users\Administrator\Downloads\Compressed\yolo_objectDetection_imagesCPU-master\yolo_objectDetection_imagesCPU-master\yolov3.weights"  # Path to the YOLOv3 weights file
names_file = r"C:\Users\Administrator\Downloads\Compressed\yolo_objectDetection_imagesCPU-master\yolo_objectDetection_imagesCPU-master\coco.names"  # Path to the COCO class names file

# Check if files exist
if not os.path.exists(cfg_file):
    print(f"Error: The file {cfg_file} does not exist.")
    exit()

if not os.path.exists(weights_file):
    print(f"Error: The file {weights_file} does not exist.")
    exit()

if not os.path.exists(names_file):
    print(f"Error: The file {names_file} does not exist.")
    exit()

# Load YOLO model
yolo = cv2.dnn.readNet(weights_file, cfg_file)

# Load class names from coco.names
classes = []
with open(names_file, "r") as file:
    classes = [line.strip() for line in file.readlines()]

# Get YOLO network output layer names
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers().flatten()]

# Color settings
colorRed = (0, 0, 255)  # Red for text
colorGreen = (0, 255, 0)  # Green for bounding boxes

# Load the image to be processed
name = r"C:\Users\Administrator\Downloads\Compressed\yolo_objectDetection_imagesCPU-master\yolo_objectDetection_imagesCPU-master\image.jpg"  # Path to the image file
img = cv2.imread(name)

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Unable to load image {name}")
    exit()

height, width, channels = img.shape

# Detect objects in the image
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
yolo.setInput(blob)
outputs = yolo.forward(output_layers)

# Initialize lists to hold detected boxes, confidences, and class ids
class_ids = []
confidences = []
boxes = []

# Loop through each output from the model
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Confidence threshold
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Get coordinates for the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Append the box, confidence, and class ID
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maxima suppression to eliminate redundant boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Convert indexes to a 1D array, if necessary
if len(indexes) > 0:
    indexes = indexes.flatten()

# Draw bounding boxes and labels on the image
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(img, (x, y), (x + w, y + h), colorGreen, 3)
        cv2.putText(img, label, (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 1, colorRed, 2)

# Save the result image
output_image_path = "output.jpg"
cv2.imwrite(output_image_path, img)
print(f"Object detection complete. The output image is saved as {output_image_path}")

# Optionally, display the image (commented out for non-GUI environments)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
