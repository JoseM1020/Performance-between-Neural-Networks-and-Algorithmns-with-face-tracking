#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/FaceDetect
DATA=data/FaceDetect
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/facedetect_train_lmdb \
  $DATA/face_mean.binaryproto

echo "Done."
