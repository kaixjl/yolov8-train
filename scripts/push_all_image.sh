#!/bin/bash

# run this script in parent folder which contains folders 'docker' and 'scripts'.

bash scripts/push_detection_image.sh
bash scripts/push_segmentation_image.sh
# bash scripts/push_pose_image.sh
bash scripts/push_coco_det_image.sh
bash scripts/push_voc_det_image.sh
bash scripts/push_empty_image.sh