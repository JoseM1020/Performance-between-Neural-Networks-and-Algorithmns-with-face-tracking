import numpy as np
import argparse
import cv2
import os
from timing import Timing


ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# image = cv2.imread(args["image"])

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, 'dataC'))
images = [image for image in os.listdir(DATA_DIR) if '.jpg' in image]
images1 = images[0]
images10 = images[:10]

def detect_faces(image_path):
    image = cv2.imread('dataC/'+image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob as the input to the network
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # find the predictions confidence value
        confidence = detections[0, 0, i, 2]
        # Only show detections above the given threshold
        if confidence > args["confidence"]:
            # compute the coordinates of the bounding box for the detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box and display the confidence
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 255), 2)
    # cv2.imshow("Output", image)
    cv2.imwrite(f"./outC/detected_{os.path.splitext(image_path)[0]}.jpg", image)
    # cv2.imwrite(f"./outC/detected_{image_path}", image)


timer = Timing("1 image test")
detect_faces(images1)
timer.end_log()

timer = Timing("10 image test")
for i in images10:
    detect_faces(i)
timer.end_log()

timer = Timing("100 image test")
for i in images:
    detect_faces(i)
timer.end_log()