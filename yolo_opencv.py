import cv2
import numpy as np

import time
import sys
import os

# adapted from https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/


# Minimum probability to filter weak detections.
# a default value of 50% (0.5 ),
# you should feel free to experiment with this value.
CONFIDENCE = 0.5

# non-maxima suppression threshold with a default value of 0.5 .
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# the neural network configuration
config_path = "cfg/yolo-obj.cfg"

# the YOLO net weights file
weights_path = "weights/yolo-obj_final.weights"

# loading all the class labels (objects)
LABELS = open("data/obj.names").read().strip().split("\n")

brand_counts = {label: 0 for label in LABELS}

# generating colors for each object for later plotting
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# load the YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)


# path name to the image
path_name = "data/ford.jpg"

# arrange filenames to later save the deteceted object image
# jfif extension may cause problems, try jpg format
file_name = os.path.basename(path_name)
filename, ext = file_name.split(".")

# load the input image and get its spatial dimensions
image = cv2.imread(path_name)
h, w = image.shape[:2]

# 4D blob
# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving the our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# sets the blob as the input of the network
net.setInput(blob)

# get all the layer names
ln = net.getLayerNames()

# determine only the *output* layer names that we need from YOLO to detect
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]



# feed forward (inference) and get the network output
# measure how much it took in seconds
start = time.perf_counter()
layer_outputs = net.forward(ln)
time_took = time.perf_counter() - start
print(f"Time took: {time_took:.2f}s")

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes, confidences, class_ids = [], [], []


# boxes : The bounding boxes around the object.
# confidences : The confidence value that YOLO assigns to an object. 
# classIDs : The detected object’s class label.

# Lower confidence values indicate that the object might not be what the network thinks it is. 
# Remember from the constans above that it will filter out the objects that don’t meet the 0.5 threshold.


# loop over each of the layer outputs
for output in layer_outputs:

    # loop over each of the object detections
    for detection in output:
        # extract the class id (label) and confidence (as a probability) of
        # the current object detection
        scores = detection[5:]
        class_id = np.argmax(scores)    # the most probable class id 
        confidence = scores[class_id]
        
        # discard weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > CONFIDENCE:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")

            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # update our list of bounding box coordinates, confidences,
            # and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)



# Applying non-maxima suppression suppresses significantly overlapping bounding boxes, 
# keeping only the most confident ones.

# perform the non maximum suppression given the scores defined before
idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

# define visual content values for drawing bounding box around the detected object 
font_scale = 1
font_thickness = 1
font_color = (255, 255, 255)

thickness = 1

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():

        # extract the bounding box coordinates
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        
        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
        text = f"{LABELS[class_ids[i]]}: {confidences[i]:.2f}"
        # calculate text width & height to draw the transparent boxes as background of the text
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        text_offset_x = x
        text_offset_y = y - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
        
        # add opacity (transparency to the box)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        # now put the text (label: confidence %)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_DUPLEX,
            fontScale=font_scale, color=font_color, thickness=font_thickness)

        brand_counts[LABELS[class_ids[i]]] += 1
        
# display image  
cv2.imshow("image", image)
if cv2.waitKey(0) == ord("q"):
    pass

output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Guarda a imagem na pasta output_images
output_path = os.path.join(output_dir, filename + "_yolov4." + ext)
cv2.imwrite(output_path, image)

print(f"Imagem guardada em: {output_path}")


print("\nRelatório de contagem de carros por marca:")
for brand, count in brand_counts.items():
    if count > 0:
        print(f"{brand}: {count}")
