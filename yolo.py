'''
CPSC 393 - Final Project

File: yolo.py
Description: Performing object detection on images using YOLO (single-stage detector) pre-trained on the COCO dataset

Adopted From: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
'''
import numpy as np
import time
import cv2
import os

yoloDirectory = "yolo-coco"
directoryToSearch = "images"
minConfidence = 0.7
minThreshold = 0.3

def processImage(imagePath):
	imageToProcess = cv2.imread(imagePath)
	(imageHeight, imageWidth) = imageToProcess.shape[:2]

	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	blob = cv2.dnn.blobFromImage(imageToProcess, 1 / 255.0, (416, 416),
								 swapRB=True, crop=False)
	net.setInput(blob)
	startTime = time.time()
	layerOutputs = net.forward(ln)
	endTime = time.time()
	timeTaken = endTime-startTime

	print("[INFO] YOLO TOOK " + str(timeTaken) + "sec on " + imagePath)

	boxes = []
	classIDs = []
	confidences = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > minConfidence:
				box = detection[0:4] * np.array([imageWidth, imageHeight, imageWidth, imageHeight])
				(centerX, centerY, width, height) = box.astype("int")

				startX = int(centerX - (width / 2))
				startY = int(centerY - (height / 2))

				boxes.append([startX, startY, int(width), int(height)])
				classIDs.append(classID)
				confidences.append(float(confidence))

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, minConfidence,
							minThreshold)

	if len(idxs) > 0:
		for i in idxs.flatten():
			(startX, startY) = (boxes[i][0], boxes[i][1])
			(boxWidth, boxHeight) = (boxes[i][2], boxes[i][3])

			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(imageToProcess, (startX, startY), (startX + boxWidth, startY + boxHeight), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(imageToProcess, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX,
						0.5, color, 2)

	cv2.imshow("Image", imageToProcess)
	cv2.waitKey(0)

labelsPath = os.path.sep.join([yoloDirectory, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = os.path.sep.join([yoloDirectory, "yolov3.weights"])
configPath = os.path.sep.join([yoloDirectory, "yolov3.cfg"])

print("[INFO] loading YOLO (pretrained on COCO) from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

for file in os.listdir(directoryToSearch):
	processImage(directoryToSearch + "/" + file)