#!/bin/bash

# run this script in parent folder which contains folders 'docker' and 'scripts'.

CONTAINER_TAG=yolov8-dev
docker rm -f $CONTAINER_TAG
docker run --gpus all -t -d -v ./dataset/coco8-pose:/dataset -v ./weight/pose:/weight -v .:/root/workspace/yolov8-train --name $CONTAINER_TAG ultralytics/ultralytics
docker exec $CONTAINER_TAG bash -c "mkdir /root/sources && ln -s /usr/src/ultralytics /root/sources/yolov8"