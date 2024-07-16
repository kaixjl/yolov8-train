#!/bin/bash

# run this script in parent folder which contains folders 'docker' and 'scripts'.

IMAGE_TAG=empty-train
docker run --gpus all --ipc host -it --rm \
    -e MODEL_WORKING_MODE=1 \
    -e HP_EPOCHES=20 \
    -e HP_BATCH_SIZE=16 \
    -e HP_LEARNING_RATE=0.01 \
    -e HP_WEIGHT_DECAY=0.0005 \
    -e HP_MOMENTUN=0.937 \
    -e HP_CONFIDENCE=0.85 \
    -v ./weight/detect:/weight \
    -v ./dataset/coco8:/dataset \
    ${IMAGE_TAG} $1
