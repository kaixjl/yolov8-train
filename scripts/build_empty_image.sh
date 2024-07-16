#!/bin/bash

# run this script in parent folder which contains folders 'docker' and 'scripts'.

IMAGE_TAG=empty-train
docker rmi ${IMAGE_TAG}
docker build -f docker/Empty.Dockerfile \
    -t ${IMAGE_TAG} .