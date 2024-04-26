#!/bin/bash

# run this script in parent folder which contains folders 'docker' and 'scripts'.

IMAGE_TAG=yolov8-seg-train
docker rmi ${IMAGE_TAG}
docker build -f docker/Segmentation.Dockerfile -t ${IMAGE_TAG} .