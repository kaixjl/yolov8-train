#!/bin/bash

# run this script in parent folder which contains folders 'docker' and 'scripts'.

bash scripts/build_detection_image.sh
bash scripts/build_segmentation_image.sh
# bash scripts/build_pose_image.sh
bash scripts/build_coco_det_image.sh
bash scripts/build_voc_det_image.sh