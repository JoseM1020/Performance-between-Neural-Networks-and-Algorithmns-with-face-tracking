import os
import numpy as np
import cv2
from timing import Timing

haarcascade_path = cv2.data.haarcascades + "/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haarcascade_path)

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../CaffeImageDetection/dataC'))
images = [image for image in os.listdir(DATA_DIR) if '.jpg' in image]
images1 = images[0]
images10 = images[:10]


def detect_faces(image):
    img = cv2.imread("../CaffeImageDetection/dataC/"+image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
    cv2.imwrite(f"./outC/detected_{os.path.splitext(image)[0]}.jpg", img)


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

# cv2.imshow('img',img)
# cv2.waitKey(0)
# v2.destroyAllWindows()
