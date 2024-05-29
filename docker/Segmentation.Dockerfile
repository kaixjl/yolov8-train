FROM ultralytics/ultralytics

# pull yolov8 modified by rockchip.
RUN git clone https://github.com/airockchip/ultralytics_yolov8 /usr/src/yolov8-rk
ENV PYTHONPATH=/usr/src/yolov8-rk

ENV MODEL_DATA_TYPE=image

ENV MODEL_TRAINING=true
ENV MODEL_CONVERSION=false
ENV MODEL_INFERENCE=false

ENV USE_GPU=true

ENV HP_EPOCHES numeric:[1,)
ENV HP_BATCH_SIZE numeric:[1,)
ENV HP_LEARNING_RATE numeric:(,)
ENV HP_WEIGHT_DECAY numeric:(,)
ENV HP_MOMENTUN numeric:(,)
ENV HP_CONFIDENCE numeric:(0,1)

RUN mkdir -p /root/workspace/yolov8-train
COPY start_seg.py /root/workspace/yolov8-train
COPY yolov8n-seg.pt /root/workspace/yolov8-train
COPY yolov8n.pt /root/workspace/yolov8-train

WORKDIR /root/workspace/yolov8-train

CMD [ "python", "start_seg.py" ]

# CMD [ "/bin/bash" ]