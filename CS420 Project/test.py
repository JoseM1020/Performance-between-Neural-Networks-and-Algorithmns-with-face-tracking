import picamera

print("about to take photo")
with picamera.PiCamera() as camera:
    camera.resolution = (1280, 720)
    camera.capture("/home/pi/Desktop/output.jpg")
print("photo taken.")