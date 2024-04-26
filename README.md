# 说明

本代码中包含构建yolov8检测、分割、人体姿态镜像的文件，数据集目录结构参考Dockerfile标准-\*.pdf文件。

# 构建

可通过在根目录（包含`scripts/`, `docker/`等的目录）中调用`scripts`目录中的脚本`build_detection_image.sh`, `build_segmentation_image.sh`, `build_pose_image.sh`分别构建yolov8检测、分割、人体姿态镜像，`build_all_image.sh`可构建全部三种镜像。

# 运行

可通过在根目录（包含`scripts/`, `docker/`等的目录）中调用`scripts`目录中的脚本`run_det_in_container.sh`, `run_seg_in_container.sh`, `run_pose_in_container.sh`分别构建yolov8检测、分割、人体姿态镜像。

# 修订

参考`CHANGELOG.md`。
