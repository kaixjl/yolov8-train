#!/bin/bash

# run this script in parent folder which contains folders 'docker' and 'scripts'.

IMAGE_TAG=yolov8-det-train-voc
docker rmi ${IMAGE_TAG}
docker build -f docker/Detect-VOC.Dockerfile \
    -t ${IMAGE_TAG} .