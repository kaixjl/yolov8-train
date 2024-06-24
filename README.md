# 说明

本代码中包含构建yolov8检测、分割镜像的文件，数据集目录结构参考Dockerfile标准-\*.pdf文件。

# 构建

可通过在根目录（包含`scripts/`, `docker/`等的目录）中调用`scripts`目录中的脚本`build_detection_image.sh`, `build_segmentation_image.sh`分别构建yolov8检测、分割镜像，`build_all_image.sh`可构建全部三种镜像。

# 运行

可通过在根目录（包含`scripts/`, `docker/`等的目录）中调用`scripts`目录中的脚本`run_det_in_container.sh`, `run_seg_in_container.sh`分别构建yolov8检测、分割镜像。

镜像需要输入如下环境变量：

| 环境变量 | 必填/可选 | 说明 |
| -- | -- | -- |
| MODEL_WORKING_MODE | 必填 | 整数；指示镜像的运行模式；可接受的值包括“1（训练模式），4（手动验证），5（质检功能）” |

在`MODEL_WORKING_MODE==1`时，需要如下环境变量：

| 环境变量 | 必填/可选 | 说明 |
| -- | -- | -- |
| HP_EPOCHES | 必填 | 整数，大于0 |
| HP_BATCH_SIZE | 必填 | 整数，大于0 |
| HP_LEARNING_RATE | 必填 | 浮点数 |
| HP_WEIGHT_DECAY | 必填 | 浮点数 |
| HP_MOMENTUN | 必填 | 浮点数 |

同时需要挂载如下目录：

| 镜像内目录 | 说明 |
| -- | -- |
| /weight | 模型文件目录，具体参考`Dockerfile标准-*.pdf` |
| /dataset | 数据集目录，具体参考`Dockerfile标准-*.pdf` |

在`MODEL_WORKING_MODE==4`时，需要如下环境变量：

| 环境变量 | 必填/可选 | 说明 |
| -- | -- | -- |
| HP_CONFIDENCE | 必填 | 浮点数，取值范围为(0,1) |

同时需要挂载如下目录：

| 镜像内目录 | 说明 |
| -- | -- |
| /weight | 模型文件目录，具体参考`Dockerfile标准-*.pdf` |
| /dataset | 数据集目录，具体参考`Dockerfile标准-*.pdf` |

在`MODEL_WORKING_MODE==5`时，需要挂载如下目录：

| 镜像内目录 | 说明 |
| -- | -- |
| /weight/output | 模型文件目录，具体参考`Dockerfile标准-*.pdf` |
| /dataset | 数据集目录，具体参考`Dockerfile标准-*.pdf` |

## 手动验证模式

## 质检模式

# 修订

参考`CHANGELOG.md`。
