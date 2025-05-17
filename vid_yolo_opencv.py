import cv2
import numpy as np
from sort import Sort

import time
import sys
import os

# adapted from https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

# Minimum probability to filter weak detections.
# a default value of 50% (0.5 ),
# but you should feel free to experiment with this value.
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


# load the YOLO object detector trained on the custom dataset (10 classes)
# and determine only the *output* layer names that is needed from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


# the default saving video file extension for this code is avi (for Windows)
# this code is tested with mp4 extension videos

# change path name to experiment with videos
path_name = "data/bentley_chrysler.mp4"
output_path = "output_video/bentley_chrysler.avi"


# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(path_name)
writer = None
(W, H) = (None, None)


tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
unique_cars = {} 

while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

    dets = []
    det_brands = []
    det_confs = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            dets.append([x, y, x + w, y + h, confidences[i]])
            det_brands.append(classIDs[i])
            det_confs.append(confidences[i])

    # Garante que dets é sempre um array numpy
    if len(dets) > 0:
        dets_np = np.array(dets)
    else:
        dets_np = np.empty((0, 5))

    tracks = tracker.update(dets_np)
    # Para cada track, associa a detection mais próxima (pelo centro)
    for track in tracks:
        track_id = int(track[4])
        x1, y1, x2, y2 = map(int, track[:4])
        # Encontra a detection mais próxima deste track
        min_dist = float('inf')
        best_idx = -1
        for idx, det in enumerate(dets):
            dx1, dy1, dx2, dy2, _ = det
            center_det = ((dx1 + dx2) / 2, (dy1 + dy2) / 2)
            center_track = ((x1 + x2) / 2, (y1 + y2) / 2)
            dist = np.linalg.norm(np.array(center_det) - np.array(center_track))
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        # Só associa se ainda não existir
        if track_id not in unique_cars and best_idx != -1:
            unique_cars[track_id] = (LABELS[det_brands[best_idx]], det_confs[best_idx])
        brand, conf = unique_cars.get(track_id, ("Desconhecido", 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{brand} {conf:.2f} ID:{track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Escreve o frame no vídeo de saída
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()

# Relatório final
from collections import Counter
# Conta só a marca, ignorando a confiança
brand_counts = Counter([brand for brand, conf in unique_cars.values()])
print("\nRelatório de contagem de carros por marca:")
for brand, count in brand_counts.items():
    print(f"{brand}: {count}")
