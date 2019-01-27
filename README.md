# Object Detection In Media Using YoloV3

#### CPSC 393 Machine Learning Final Project

Carlos Amesquita, Everett Yee, Keanu Kauhi-Correia

In order for YOLOv3 to work, the weights file must be put into yolo-coco folder. It can be found [here](https://pjreddie.com/media/files/yolov3.weights)

## Instructions for Replication
Directory Structure should look like this:

root
├── annotations
├── images
├── output
├── yolo-coco
│   ├── coco.names
│   ├── yolov3.cfg
│   └── yolov3.weights
├── accuracyTracker.py
└── yolo.py

#### Configurations
There are two files that have a configuration.
- yolo.py (lines 18-25)
- accuracyTracker.py (lines 14-18)

1. If you have annotations for truth comparisons, put them in the annotations folder and configure accuracyTracker.py
2. Put your images to be labeled in the /images folder
3. Run yolo.py, then accuracyTracker.py (if wanting to find accuracy)

