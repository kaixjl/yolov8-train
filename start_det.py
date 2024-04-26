# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from copy import copy
from pathlib import Path
import os

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from ultralytics.engine.validator import BaseValidator
from ultralytics.engine.trainer import BaseTrainer
from ultralytics import RTDETR, YOLO


epoch = 0
epochs = 0
loss = 0
# box_loss, cls_loss, dfl_loss = 0
mode = "train"
model = None # type: YOLO


def on_train_epoch_end(trainer):
    # type: (BaseTrainer) -> None

    global epoch, loss, mode

    epoch = trainer.epoch
    loss = trainer.loss.detach().cpu().item()
    # box_loss, cls_loss, dfl_loss = trainer.loss_items.detach().cpu()


def on_val_end(validator):
    # type: (BaseValidator) -> None

    global epoch, loss, mode

    if not (mode=="val" or mode=="train"): return

    stats = validator.get_stats()
    print("======")
    if mode == "train":
        print(f"Epoch: {epoch + 1} | train loss: {loss} | test accuracy: { stats['metrics/precision(B)'] }")
    elif mode == "val":
        print("tested: {tested}, correct: {correct}, accuracy: {accuracy:.6f}, recall: {recall:.6f}, map50: {map50:.6f}, map50_95: {map50_95:.6f}".format(
            tested=validator.seen,
            correct=validator.seen,
            accuracy=stats['metrics/precision(B)'],
            recall=stats['metrics/recall(B)'],
            map50=stats['metrics/mAP50(B)'],
            map50_95=stats['metrics/mAP50-95(B)']
        ))
    print("======")


def on_val_batch_end(validator):
    # type: (BaseValidator) -> None

    global epoch, loss, mode

    if not (mode=="val"): return

    stats = validator.get_stats()
    print("======")
    if mode == "val":
        print("tested: {tested}, correct: {correct}, accuracy: {accuracy:.6f}, recall: {recall:.6f}, map50: {map50:.6f}, map50_95: {map50_95:.6f}".format(
            tested=validator.seen,
            correct=validator.seen,
            accuracy=stats['metrics/precision(B)'],
            recall=stats['metrics/recall(B)'],
            map50=stats['metrics/mAP50(B)'],
            map50_95=stats['metrics/mAP50-95(B)']
        ))
    print("======")


def prepare_dataset_3():
    # 在/tmp/dataset目录中准备数据集，目录结构如下
    # /tmp/dataset
    #   |- [images]
    #   |  |- [train]
    #   |  |- [val]
    #   |  |- [test]
    #   |- [labels]
    #   |  |- [train]
    #   |  |- [val]
    #   |  |- [test]
    #   |- data.yaml
    os.system("rm -r /tmp/dataset")
    os.system("mkdir -p /tmp/dataset/images")
    os.system("mkdir -p /tmp/dataset/labels")
    with open("/dataset/data.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    data["path"] = "/tmp/dataset"
    data["train"] = []
    data["val"] = []
    data["test"] = []

    for s in ["train", "test"]:
        if not os.path.exists(f"/dataset/{s}/images") or not os.path.exists(f"/dataset/{s}/annotations"):
            continue
        os.system(f"mkdir -p /tmp/dataset/images/{s}")
        os.system(f"ln -s /dataset/{s}/annotations /tmp/dataset/labels/{s}") # 创建labels目录的软链接
        labels = os.listdir(f"/tmp/dataset/labels/{s}")
        ext = os.listdir(f"/dataset/{s}/images")[0][-4:]
        images = [f"{it[:-4]}{ext}" for it in labels]
        for it in images:
            if os.path.exists(f"/dataset/{s}/images/{it}"):
                os.system(f"ln -s /dataset/{s}/images/{it} /tmp/dataset/images/{s}/") # 创建图像文件的软链接
        data[s] = f"images/{s}"
    while True:
        if not os.path.exists(f"/dataset/validation/images") or not os.path.exists(f"/dataset/validation/annotations"):
            continue
        os.system(f"mkdir -p /tmp/dataset/images/val")
        os.system(f"ln -s /dataset/validation/annotations /tmp/dataset/labels/val")
        labels = os.listdir(f"/tmp/dataset/labels/val")
        ext = os.listdir(f"/dataset/validation/images")[0][-4:]
        images = [f"{it[:-4]}{ext}" for it in labels]
        for it in images:
            if os.path.exists(f"/dataset/validation/images/{it}"):
                os.system(f"ln -s /dataset/validation/images/{it} /tmp/dataset/images/val/")
        data["val"] = f"images/val"
        break

    with open("/tmp/dataset/data.yaml", "w") as f:
        yaml.dump(data, f)


def create_model():
    global model

    print("Start creating model.")
    pretrained = "yolov8n.pt"
    if os.path.exists("/weight/input") and os.path.isdir("/weight/input"):
        pretraineds = os.listdir("/weight/input")
        if len(pretraineds) > 0:
            pretrained = os.path.join("/weight/input", pretraineds[0])
    print(f"Use pretrained file '{pretrained}'")

    model = YOLO(pretrained)
    model.add_callback("on_val_end", on_val_end)
    model.add_callback("on_val_batch_end", on_val_batch_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)


def train():
    global mode, model

    HP_EPOCHES = int(os.environ["HP_EPOCHES"])
    HP_BATCH_SIZE = int(os.environ["HP_BATCH_SIZE"])
    HP_LEARNING_RATE = float(os.environ["HP_LEARNING_RATE"])
    HP_WEIGHT_DECAY = float(os.environ["HP_WEIGHT_DECAY"])
    HP_MOMENTUN = float(os.environ["HP_MOMENTUN"])

    mode = "train"
    print("Start training model.")
    os.system("rm -r /tmp/run/*")
    os.system("mkdir -p /tmp/run")
    model.train(data="/tmp/dataset/data.yaml",
                project="/tmp/run",
                name="train",
                epochs=HP_EPOCHES,
                batch=HP_BATCH_SIZE,
                lr0=HP_LEARNING_RATE,
                weight_decay=HP_WEIGHT_DECAY,
                momentum=HP_MOMENTUN)
    # model.export()
    os.system("mkdir -p /weight/output")
    os.system("cp /tmp/run/train/weights/best.pt /weight/output")

def val():
    global mode, model

    mode = "val"
    print("Start validating model.")
    total = len(os.listdir("/tmp/dataset/labels/test"))
    print(f"Total sample number: {total}")
    model.val(split="test")


def training_mode():
    prepare_dataset_3()
    create_model()
    train()
    val()


def user_validation_mode():
    import sys
    WEIGHT = "/weight/output/best.pt"
    INPUTS_DIR = "/dataset/user_validation/images"
    OUTPUTS_DIR = "/dataset/user_validation/output"
    HP_CONFIDENCE = float(os.environ["HP_CONFIDENCE"])

    if not os.path.exists(WEIGHT):
        print("Cannot find output weight file '/weight/output/best.pt'", file=sys.stderr)
        exit(1)

    os.system(f"rm {OUTPUTS_DIR}/*")
    os.system(f"mkdir -p {OUTPUTS_DIR}")
    model = YOLO(WEIGHT)
    results = model(INPUTS_DIR, conf=HP_CONFIDENCE)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.save(filename=os.path.join(OUTPUTS_DIR, os.path.basename(result.path)))  # save to disk
        # 结果文件中，每行为一个对象，每列分别为：
        #     1：类别
        #     2：置信度
        #     3-6 ：normalized (x_center,y_center,w,h)
        with open(os.path.join(OUTPUTS_DIR, os.path.basename(result.path)[:-4] + ".txt"), 'w') as f:
            nobj = len(boxes.cls)
            for i in range(nobj):
                f.write("{:d} {:.6f} {}\n".format(
                    int(boxes.cls[i]),
                    boxes.conf[i],
                    " ".join(["{:.6f}".format(idx) for idx in boxes.xywhn[i].flatten()])))


def main():
    running_mode = int(os.environ["MODEL_WORKING_MODE"])
    if running_mode == 1:
        print("Run in training mode.")
        training_mode()
    elif running_mode == 4:
        print("Run in user validation mode.")
        user_validation_mode()
    else:
        print("Image cannot run in modes other than training mode and user validation mode.")


if __name__ == "__main__":
    main()
