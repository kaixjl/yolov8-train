# FROM ppyoloe1:8.0
FROM ultralytics/ultralytics

# 标志接受数据集的类型, 可以接受：image, video, pointcloud, audio。
ENV MODEL_DATA_TYPE=image

# 标志镜像提供的功能，依次为训练、模型转换、推理。
ENV MODEL_TRAINING=true
ENV MODEL_CONVERSION=false
ENV MODEL_INFERENCE=false

# 标志镜像是否使用GPU
ENV USE_GPU=true

# 声明镜像需要通过平台向用户请求哪些超参数；镜像运行后，可以直接通过这些环境变量反过来获取用户配置的超参数的值。
ENV HP_EPOCHES numeric:[1,)
ENV HP_BATCH_SIZE numeric:[1,)
ENV HP_LEARNING_RATE numeric:(,)
ENV HP_WEIGHT_DECAY numeric:(,)
ENV HP_MOMENTUN numeric:(,)
# 手动验证时使用的置信度阈值
ENV HP_CONFIDENCE numeric:(0,1)


# 将入口脚本以及其他依赖文件导入镜像。
RUN mkdir -p /root/workspace/yolov8-train
COPY start_det.py /root/workspace/yolov8-train
COPY yolov8n.pt /root/workspace/yolov8-train

WORKDIR /root/workspace/yolov8-train

CMD [ "python", "start_det.py" ]

# CMD [ "/bin/bash" ]