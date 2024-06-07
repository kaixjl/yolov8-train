#!/bin/bash

BUILD=$(date +%Y%m%d)
rm yolov8-train.*.tgz yolov8-train.tgz
tar zcf yolov8-train.$BUILD.tgz --exclude=yolov8n-pose.pt --exclude=dataset/coco8-pose --exclude=Pose.Dockerfile --exclude=start_pose.py --exclude=build_pose_image.sh --exclude=run_pose_in_container.sh start_*.py yolov8n*.pt README.md CHANGELOG.md docker/ scripts/ dataset/coco8/ dataset/coco8-seg/ dataset/coco8-pose/ Dockerfile*.pdf