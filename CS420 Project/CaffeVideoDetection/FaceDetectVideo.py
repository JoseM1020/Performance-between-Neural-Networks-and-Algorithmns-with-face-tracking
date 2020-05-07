# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from timing import Timing

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load the model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# create the video stream and allow the camera to start
print("[INFO] starting video stream...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

frame_count = 0
while True:
    # Start timer
    timer = Timing(f"Frame {frame_count} timing")
    # grab the current frame of video and resize it to 400px wide
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # convert to blob as in the FaceDetect for images
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # input the blog into the neural net and get the resulting detections.
    net.setInput(blob)
    detections = net.forward()

    # go through each detection
    for i in range(0, detections.shape[2]):
        # find the predictions confidence value
        confidence = detections[0, 0, i, 2]
        # Only show detections above the given threshold
        if confidence < args["confidence"]:
            continue
        # compute the coordinates of the bounding box for the detection
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box and display the confidence
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 255), 2)

    # show the detected frame

    cv2.imshow("Frame", frame)
    timer.end_log()
    frame_count = frame_count+1
    # if the `q` key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
# shutdown code
cv2.destroyAllWindows()
vs.stop()
