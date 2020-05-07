from imutils.video import VideoStream
import imutils
import time
import cv2
from timing import Timing

print("[INFO] loading cascade file stream...")
haarcascade_path = cv2.data.haarcascades + "/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haarcascade_path)
print("[INFO] starting video stream...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

frame_count = 0
while True:
    timer = Timing(f"Frame {frame_count} timing")
    # grab the current frame of video and resize it to 400px wide
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
    cv2.imshow("Frame", frame)
    timer.end_log()
    frame_count = frame_count + 1
    # if the `q` key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
