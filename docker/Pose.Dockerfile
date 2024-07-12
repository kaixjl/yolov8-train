FROM ultralytics/ultralytics

# pull yolov8 modified by rockchip.
RUN git clone https://github.com/airockchip/ultralytics_yolov8 /usr/src/yolov8-rk
ENV PYTHONPATH=/usr/src/yolov8-rk

ENV MODEL_DATA_TYPE=image 

ENV MODEL_TRAINING=true
ENV MODEL_CONVERSION=false
ENV MODEL_INFERENCE=false

ENV USE_GPU=true

ENV HP_EPOCHES numeric:[1,):100
ENV HP_BATCH_SIZE numeric:[1,):16
ENV HP_LEARNING_RATE numeric:(,):0.01
ENV HP_WEIGHT_DECAY numeric:(,):0.0005
ENV HP_MOMENTUN numeric:(,):0.937
ENV HP_CONFIDENCE numeric:(0,1):0.7


# 镜像配置
ENV IMGCFG_UV_OUTPUT_ANNOTATED_IMAGE=false


RUN mkdir -p /root/workspace/yolov8-train
COPY start_pose.py /root/workspace/yolov8-train
COPY yolov8n-pose.pt /root/workspace/yolov8-train
COPY yolov8n.pt /root/workspace/yolov8-train

WORKDIR /root/workspace/yolov8-train

CMD [ "python", "start_pose.py" ]

# CMD [ "/bin/bash" ]