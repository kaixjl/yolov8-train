# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from ultralytics.engine.validator import BaseValidator
from ultralytics.engine.trainer import BaseTrainer
from ultralytics import RTDETR, YOLO
from ultralytics.cfg import TASK2DATA
from ultralytics.data.build import load_inference_source
from ultralytics.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_PATH,
    LINUX,
    MACOS,
    ONLINE,
    ROOT,
    WEIGHTS_DIR,
    WINDOWS,
    Retry,
    checks,
    is_dir_writeable,
    IS_RASPBERRYPI,
)
from ultralytics.utils.downloads import download
from ultralytics.utils.torch_utils import TORCH_1_9, TORCH_1_13

MODEL = WEIGHTS_DIR / "path with spaces" / "yolov8n.pt"  # test spaces in path
CFG = "yolov8n.yaml"
SOURCE = ASSETS / "bus.jpg"
TMP = (ROOT / "../tests/tmp").resolve()  # temp directory for test files
IS_TMP_WRITEABLE = is_dir_writeable(TMP)

epoch = 0
epochs = 0

def on_train_epoch_end(trainer):
    # type: (BaseTrainer) -> None

    global epoch

    epoch = trainer.epoch

def on_val_end(validator):
    # type: (BaseValidator) -> None

    global epoch

    print("======")
    print(f"Epoch: {epoch}")
    print("Validator Stats:", validator.get_stats())
    print("Seen:", validator.seen)
    print("======")

def train():
    model = YOLO("./weight/input/pretrained.pt")
    model.add_callback("on_val_end", on_val_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    # model.train(data="coco128.yaml", epochs=50, imgsz=32, cache="ram", copy_paste=0.5, mixup=0.5, name=0)
    model.train(data="coco128.yaml", epochs=50)
    model(SOURCE)

def main():
    train()

if __name__ == "__main__":
    main()
