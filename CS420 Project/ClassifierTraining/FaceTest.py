import os

import numpy as np
import cv2

haarcascades_path = cv2.data.haarcascades + "/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haarcascades_path)

test_cascade = cv2.CascadeClassifier('./ClassifierV2/cascade.xml')

img = cv2.imread('faceTest2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

#cv2.imshow('img',img)

#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imwrite("./out/default_result2.jpg", img)

img = cv2.imread('faceTest2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = test_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

cv2.imwrite("./out/test_result2.jpg", img)