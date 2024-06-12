#!/bin/bash

# run this script in parent folder which contains folders 'docker' and 'scripts'.

IMAGE_TAG=yolov8-det-train
docker rmi registry.cn-hangzhou.aliyuncs.com/kaixjl/aiserver:${IMAGE_TAG}
docker tag ${IMAGE_TAG} registry.cn-hangzhou.aliyuncs.com/kaixjl/aiserver:${IMAGE_TAG}
docker push registry.cn-hangzhou.aliyuncs.com/kaixjl/aiserver:${IMAGE_TAG}