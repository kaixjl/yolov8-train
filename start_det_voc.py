# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import namedtuple
import contextlib
from copy import copy
from pathlib import Path
import os
import sys

import cv2
import numpy as np
import torch
import yaml
import json
from PIL import Image
from xml.etree import ElementTree as et

from ultralytics.engine.validator import BaseValidator
from ultralytics.engine.trainer import BaseTrainer
from ultralytics import RTDETR, YOLO
from ultralytics.engine.results import Results


epoch = 0
epochs = 0
loss = 0
# box_loss, cls_loss, dfl_loss = 0
mode = "train"
model = None # type: YOLO
running_mode = 0


def on_fit_epoch_end(trainer):
    # type: (BaseTrainer) -> None

    global epoch, loss, mode

    epoch = trainer.epoch
    loss = trainer.loss.detach().cpu().item()
    # box_loss, cls_loss, dfl_loss = trainer.loss_items.detach().cpu()
    print("===on_fit_epoch_end===")
    if mode == "train":
        print(f"Epoch: {epoch + 1} | train loss: {loss} | test accuracy: { trainer.metrics['metrics/precision(B)'] }")
    print("=======")


def on_val_end(validator):
    # type: (BaseValidator) -> None

    global epoch, loss, mode

    if not (mode=="val" or mode=="train"): return

    stats = validator.get_stats()
    print("===on_val_end===")
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


def on_val_batch_end(validator):
    # type: (BaseValidator) -> None

    global epoch, loss, mode

    if not (mode=="val"): return

    stats = validator.get_stats()
    print("===on_val_batch_end===")
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


def parse_voc_annotations(anno_dir):
    CatInfo = namedtuple("CatInfo", ["name"])
    AnnoInfo = namedtuple("AnnoInfo", ["bbox", "category"])
    ImageInfo = namedtuple("ImageInfo", ["filename", "width", "height", "annotations"])

    images = [] # type: list[ImageInfo]
    files = os.listdir(anno_dir)
    for it in files:
        path = os.path.join(anno_dir, it)
        tree = et.parse(path)
        root = tree.getroot()
        filename = root.find("filename").text
        size = root.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)

        annotations = [] # type: list[AnnoInfo]
        for object in root.iter("object"):
            name = object.find("name").text
            bndbox = object.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            bwidth = xmax - xmin
            bheight = ymax - ymin
            annotations.append(AnnoInfo([xmin, ymin, bwidth, bheight], CatInfo(name)))

        images.append(ImageInfo(filename=filename,
                                width=width,
                                height=height,
                                annotations=annotations))
    return images


def prepare_dataset_3(ssplits=["train", "test", "validation"], dsplits=["train", "test", "val"]):
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

    print("===prepare_dataset_3===")
    print("Preparing dataset...")

    def check():
        if not os.path.exists("/dataset/data.yaml"):
            print("Cannot find /dataset/data.yaml.")
            return False
        for s in ssplits:
            if not os.path.exists(f"/dataset/{s}/images") or not os.path.exists(f"/dataset/{s}/annotations"):
                print(f"Cannot find /dataset/{s}/images or /dataset/{s}/annotations.")
                return False
        return True
    
    if not check(): return False

    os.system("rm -rf /tmp/dataset")
    os.system("mkdir -p /tmp/dataset/images")
    os.system("mkdir -p /tmp/dataset/labels")
    with open("/dataset/data.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    data["path"] = "/tmp/dataset"
    data["train"] = "images/train"
    data["val"] = "images/val"
    data["test"] = "images/test"

    for s, t in zip(ssplits, dsplits):
        os.system(f"mkdir -p /tmp/dataset/images/{t}")
        os.system(f"mkdir -p /tmp/dataset/labels/{t}") # 创建labels目录的软链接
        anno_dir = f"/dataset/{s}/annotations"
        annodata = parse_voc_annotations(anno_dir)
        name_ids = { v:k for k,v in data["names"].items() }
        for it in annodata:
            with open(os.path.join(f"/tmp/dataset/labels/{t}", os.path.splitext(it.filename)[0]+".txt"), "w") as f:
                for it2 in it.annotations:
                    f.write("{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n".format(
                        cls=name_ids[it2.category.name],
                        x=it2.bbox[0]/it.width,
                        y=it2.bbox[1]/it.height,
                        w=it2.bbox[2]/it.width,
                        h=it2.bbox[3]/it.height,
                    ))
        labels = os.listdir(f"/tmp/dataset/labels/{t}")
        imagefiles = os.listdir(f"/dataset/{s}/images")
        exts = dict([os.path.splitext(it) for it in imagefiles])
        images = [f"{it[:-4]}{exts[it[:-4]]}" for it in labels]
        n_images = 0
        for it in images:
            if os.path.exists(f"/dataset/{s}/images/{it}"):
                os.system(f"ln -s /dataset/{s}/images/{it} /tmp/dataset/images/{t}/") # 创建图像文件的软链接
                n_images += 1
        data[t] = f"images/{t}"
        print(f"Got {n_images} images and {len(labels)} labels in {t} split.")


    with open("/tmp/dataset/data.yaml", "w") as f:
        yaml.dump(data, f)
    
    return True


def create_model():
    global model

    print("===create_model===")
    print("Start creating model.")

    model_path = ""
    if running_mode == 1:
        pretrained = "yolov8n.pt"
        if os.path.exists("/weight/input") and os.path.isdir("/weight/input"):
            pretraineds = os.listdir("/weight/input")
            if len(pretraineds) > 0:
                pretrained = os.path.join("/weight/input", pretraineds[0])
        model_path = pretrained
        print(f"Use pretrained file '{pretrained}'")
    elif running_mode == 4 or running_mode == 5:
        if not os.path.exists("/weight/output"):
            print("Cannot find output directory '/weight/output'", file=sys.stderr)
            exit(1)

        model_paths = os.listdir("/weight/output")
        model_paths = [ it for it in model_paths if it.endswith(".pt") ]
        if len(model_paths) <= 0:
            print("Cannot find output weight file in '/weight/output'", file=sys.stderr)
            exit(1)

        model_path = os.path.join("/weight/output", model_paths[0])
        print(f"Use weight file '{model_path}'")

    model = YOLO(model_path)
    model.add_callback("on_val_end", on_val_end)
    model.add_callback("on_val_batch_end", on_val_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def train():
    global mode, model

    HP_EPOCHES = int(os.environ["HP_EPOCHES"])
    HP_BATCH_SIZE = int(os.environ["HP_BATCH_SIZE"])
    HP_LEARNING_RATE = float(os.environ["HP_LEARNING_RATE"])
    HP_WEIGHT_DECAY = float(os.environ["HP_WEIGHT_DECAY"])
    HP_MOMENTUN = float(os.environ["HP_MOMENTUN"])
    HP_PATIENCE = int(os.environ["HP_PATIENCE"])

    print("===train===")
    mode = "train"
    print("Start training model.")
    os.system("rm -rf /tmp/run/*")
    os.system("mkdir -p /tmp/run")
    model.train(data="/tmp/dataset/data.yaml",
                project="/tmp/run",
                name="train",
                epochs=HP_EPOCHES,
                batch=HP_BATCH_SIZE,
                lr0=HP_LEARNING_RATE,
                weight_decay=HP_WEIGHT_DECAY,
                momentum=HP_MOMENTUN,
                patience=HP_PATIENCE)
    model.export()
    os.system("mkdir -p /weight/output")
    os.system("cp /tmp/run/train/weights/best.pt /weight/output")
    os.system("cp /tmp/run/train/weights/best.onnx /weight/output")

def val():
    global mode, model

    mode = "val"
    print("Start validating model.")
    total = len(os.listdir("/tmp/dataset/labels/test"))
    print(f"Total sample number: {total}")
    model.val(split="test")


def user_validation():
    global model
    INPUTS_DIR = "/dataset/user_validation/images"
    OUTPUTS_DIR = "/dataset/user_validation/output"
    HP_CONFIDENCE = float(os.environ["HP_CONFIDENCE"])
    IMGCFG_UV_OUTPUT_ANNOTATED_IMAGE = True if os.environ["IMGCFG_UV_OUTPUT_ANNOTATED_IMAGE"].upper() == "TRUE" else False

    os.system(f"rm {OUTPUTS_DIR}/*")
    os.system(f"mkdir -p {OUTPUTS_DIR}")
    results = model(INPUTS_DIR, conf=HP_CONFIDENCE) # type: list[Results]
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        if IMGCFG_UV_OUTPUT_ANNOTATED_IMAGE:
            cv2.imwrite(os.path.join(OUTPUTS_DIR, os.path.basename(result.path)), result.plot())
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


def training_mode():
    if not prepare_dataset_3(): exit(1)
    create_model()
    train()
    val()


def user_validation_mode():
    create_model()
    user_validation()


def validation_mode():
    """
    running in validation mode.
    """
    if not prepare_dataset_3(ssplits=["test", "test", "test"], dsplits=["train", "val", "test"]): exit(1)
    create_model()
    val()


def main():
    global running_mode
    running_mode = int(os.environ["MODEL_WORKING_MODE"])
    if running_mode == 1:
        print("Run in training mode.")
        training_mode()
    elif running_mode == 4:
        print("Run in user validation mode.")
        user_validation_mode()
    elif running_mode == 5:
        print("Run in validation mode.")
        validation_mode()
    else:
        print("Image cannot run in modes other than training mode and user validation mode.")


if __name__ == "__main__":
    main()
