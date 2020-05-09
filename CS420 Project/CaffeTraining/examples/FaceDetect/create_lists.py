import numpy as np
import os
 
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../data/FaceDetect/train'))
TXT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../data/FaceDetect'))
 
face_images = [image for image in os.listdir(DATA_DIR) if '.ppm' in image]
neg_images = [image for image in os.listdir(DATA_DIR) if '.jpg' in image]
 
face_train = face_images[:int(len(face_images)*0.7)]
face_test = face_images[int(len(face_images)*0.7):]
 
neg_train = neg_images[:int(len(neg_images)*0.7)]
neg_test = neg_images[int(len(neg_images)*0.7):]
 
with open('{}/train.txt'.format(TXT_DIR), 'w') as f:
    for image in face_train:
        f.write('{} 1\n'.format(image))
    for image in neg_train:
        f.write('{} 0\n'.format(image))
 
with open('{}/test.txt'.format(TXT_DIR), 'w') as f:
    for image in face_test:
        f.write('{} 1\n'.format(image))
    for image in neg_test:
        f.write('{} 0\n'.format(image))
