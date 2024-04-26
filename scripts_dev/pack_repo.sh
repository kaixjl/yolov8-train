#!/bin/bash

rm yolov8-train.tgz
tar zcf yolov8-train.tgz start_*.py yolov8n*.pt README.md CHANGELOG.md docker/ scripts/ dataset/coco8/ dataset/coco8-seg/ dataset/coco8-pose/ Dockerfile*.pdf