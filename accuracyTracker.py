'''
CPSC 393 - Final Project
Names: Carlos Amezquita, Everett Yee, Keanu Kauhi-Correia

File: accuracyTracker.py
Description: Take in annotation data from the model and truth set, and output a final accuracy
on how the model performed
'''

import json
import numpy as np


### CONFIG
threshold = 0.05 # Tolerated pixel % difference in bounding box labels for classifying an annotation as accurate
truthDataFile = "annotations/instances_val2017.json" # Path to file that contains truth data for dataset, currently only supports COCO format
modelResultsFile = "finalProject/1Results.txt" # Path to file that contains label data from yolo.py program
### END CONFIG

def getMaxDifference(box1, box2, imageWidth, imageHeight):
    differences = np.array(box2) - np.array(box1)
    differences[0] = differences[0] / imageWidth
    differences[1] = differences[1] / imageHeight
    differences[2] = differences[2] / imageWidth
    differences[3] = differences[3] / imageHeight

    return max(differences)

def readFile(filePath):
    handle = open(filePath,"r")
    content = handle.read()
    handle.close()
    return content

def readResults():
    results = {}
    with open(modelResultsFile) as topo_file:
        for line in topo_file:
            line = line.replace('\n','')
            line = json.loads(line)
            results.update(line)
    return results

print("[INFO] Loading annotation data from disk...")
imageInfo = json.loads(readFile(truthDataFile))['images']
annotations = json.loads(readFile(truthDataFile))['annotations']
resultsToCompare = readResults()

print("[INFO] Building indexes...")
#Build index of images with their dimensions, speeds up lookup later
bufferImageInfo = {}
for image in imageInfo:
    bufferImageInfo.update({image['id'] : {"height" : image['height'], "width" : image['width']}})
imageInfo = bufferImageInfo

#Since our yolo.py file writes the full file name as the key, we change that to just the image id in order to
# match the format of the coco annotations
bufferResults = {}
amountOfLabeledObjects = 0
for imageId in resultsToCompare:
    amountOfLabeledObjects += resultsToCompare[imageId][len(resultsToCompare[imageId])-1]
    strippedImageId = int(imageId.lstrip('0').replace(".jpg", "")) #Take out extra leading 0s and file extension, leaves us with the actual image ID
    bufferResults.update({strippedImageId : resultsToCompare[imageId]})
resultsToCompare = bufferResults

#Compare truth data with output from our YOLOv3 model
for annotation in annotations:
    imageId = annotation['image_id']
    imageHeight,imageWidth = imageInfo[imageId]['height'],imageInfo[imageId]['width']

    for box in resultsToCompare[imageId]: #Search the results for the image, then delete the box list from each image dict if it's at or lowert than the threshold
        if type(box) == list:
            if getMaxDifference(box, annotation['bbox'], imageWidth, imageHeight) <= threshold:
                resultsToCompare[imageId].remove(box)

#Get amount of leftover boxes still in resultsToCompare
inaccurateAnnoatations = 0
for result,bboxes in resultsToCompare.items():
    if(len(bboxes)) > 1:
        inaccurateAnnoatations += len(bboxes)-1 #any leftover boxes is an inaccurate annotation

errorRate = inaccurateAnnoatations/amountOfLabeledObjects
accuracy = 1-errorRate

print(str(amountOfLabeledObjects) + " labeled objects found")
print(str(inaccurateAnnoatations) + " of them were misrepresented")
print("Accuracy of model: " + str(round(accuracy*100,1)) + "%")


