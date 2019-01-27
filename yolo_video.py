'''
CPSC 393 - Final Project
Names: Carlos Amezquita, Everett Yee, Keanu Kauhi-Correia
BONUS FILE

File: yolo_video.py
Description: Performing object detection on videos using YOLO (single-stage detector) pre-trained on the COCO dataset

Adopted From: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
'''
import numpy as np
import imutils
import time
import cv2
import os

yoloDirectory = "yolo-coco"
directoryToSearch = "videos"
minConfidence = 0.7
minThreshold = 0.3

def processVideo(videoPath,outputVideoPath):
	vs = cv2.VideoCapture(videoPath)
	writer = None
	(frameWidth, frameHeight) = (None, None)

	try:
		prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
			else cv2.CAP_PROP_FRAME_COUNT
		total = int(vs.get(prop))
		print("[INFO] {} total frames in video".format(total))
	except:
		print("[INFO] could not determine # of frames in video")
		print("[INFO] no approx. completion time can be provided")
		total = -1

	while True:
		(grabbed, frame) = vs.read()

		if not grabbed: # reached end of video
			break

		if frameWidth is None or frameHeight is None:
			(frameHeight, frameWidth) = frame.shape[:2]

		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
									 swapRB=True, crop=False)
		net.setInput(blob)
		startTime = time.time()
		layerOutputs = net.forward(ln)
		endTime = time.time()

		boxes = []
		classIDs = []
		confidences = []

		for output in layerOutputs:
			for detection in output:
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				if confidence > minConfidence:
					box = detection[0:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
					(centerX, centerY, width, height) = box.astype("int")

					startX = int(centerX - (width / 2))
					startY = int(centerY - (height / 2))

					boxes.append([startX, startY, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		idxs = cv2.dnn.NMSBoxes(boxes, confidences, minConfidence,
								minThreshold)

		if len(idxs) > 0:
			for i in idxs.flatten():
				(startX, startY) = (boxes[i][0], boxes[i][1])
				(boxWidth, boxHeight) = (boxes[i][2], boxes[i][3])

				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(frame, (startX, startY), (startX + boxWidth, startY + boxHeight), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]],
										   confidences[i])
				cv2.putText(frame, text, (startX, startY - 5),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

		if writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(outputVideoPath, fourcc, 30,
									 (frame.shape[1], frame.shape[0]), True)

			if total > 0:
				timeTaken = (endTime - startTime)
				print("[INFO] single frame took {:.4f} seconds".format(timeTaken))
				print("[INFO] estimated total time to finish: {:.4f}".format(
					timeTaken * total))

		writer.write(frame)

	print("[INFO] cleaning up...")
	writer.release()
	vs.release()

labelsPath = os.path.sep.join([yoloDirectory, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = os.path.sep.join([yoloDirectory, "yolov3.weights"])
configPath = os.path.sep.join([yoloDirectory, "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

for file in os.listdir(directoryToSearch):
	outputFileName = file[:len(file)-3] + 'avi'
	processVideo(directoryToSearch + "/" + file,"output/" + outputFileName)