import json
import numpy as np

threshold = 0.05
annotationsFile = "annotations/instances_val2017.json"
resultsFile = "finalProject/1Results.txt"

def getMaxDifference(box1, box2, imageWidth, imageHeight):
    maxDiffWidth = threshold*imageWidth
    maxDiffHeight = threshold*imageHeight

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
    with open(resultsFile) as topo_file:
        for line in topo_file:
            line = line.replace('\n','')
            line = json.loads(line)
            results.update(line)
    return results

imageInfo = json.loads(readFile(annotationsFile))['images']
annotations = json.loads(readFile(annotationsFile))['annotations']
resultsToCompare = readResults()

#Build index of images with their dimensions
bufferImageInfo = {}
for image in imageInfo:
    bufferImageInfo.update({image['id'] : {"height" : image['height'], "width" : image['width']}})
imageInfo = bufferImageInfo

#Since the yolo file writes the full file name as the key, we change that to just the image id to match the coco annotations
bufferResults = {}
amountOfLabeledObjects = 0
for imageId in resultsToCompare:
    amountOfLabeledObjects += resultsToCompare[imageId][len(resultsToCompare[imageId])-1]
    strippedImageId = int(imageId.lstrip('0').replace(".jpg", ""))
    bufferResults.update({strippedImageId : resultsToCompare[imageId]})
resultsToCompare = bufferResults

for annotation in annotations:
    imageId = annotation['image_id']
    imageHeight,imageWidth = imageInfo[imageId]['height'],imageInfo[imageId]['width']

    for box in resultsToCompare[imageId]:
        if type(box) == list:
            if getMaxDifference(box, annotation['bbox'], imageWidth, imageHeight) <= threshold:
                resultsToCompare[imageId].remove(box)

inaccurateAnnoatations = 0
for result,bboxes in resultsToCompare.items():
    if(len(bboxes)) > 1:
        inaccurateAnnoatations += len(bboxes)-1

errorRate = inaccurateAnnoatations/amountOfLabeledObjects
accuracy = 1-errorRate
print("Accuracy of model: " + str(round(accuracy*100,1)) + "%")


