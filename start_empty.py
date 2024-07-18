# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import sys


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


def prepare_dataset_3(ssplits=["train", "test", "validation"], dsplits=["train", "test", "val"]):
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
    
    return True


def train():
    HP_EPOCHES = int(os.environ["HP_EPOCHES"])
    HP_BATCH_SIZE = int(os.environ["HP_BATCH_SIZE"])
    HP_LEARNING_RATE = float(os.environ["HP_LEARNING_RATE"])
    HP_WEIGHT_DECAY = float(os.environ["HP_WEIGHT_DECAY"])
    HP_MOMENTUN = float(os.environ["HP_MOMENTUN"])
    HP_PATIENCE = int(os.environ["HP_PATIENCE"])

    print("===train===")    
    
    for i in range(HP_EPOCHES):
        epoch = i
        loss = HP_EPOCHES/(i+1)
        acc = i/HP_EPOCHES
        print("Epoch: {epoch} | train loss: {loss:.6f} | test accuracy: {acc:.6f}".format(
            epoch=epoch + 1,
            loss=loss,
            acc=acc,
        ))
        
    
    os.system("mkdir -p /weight/output")
    with open("/weight/output/output_weight.txt", 'w') as f:
        f.write("Output traiend-weight here.\n")

def val():
    total = 100
    print(f"Total sample number: {total}")
    for i in range(total):
        print("tested: {tested}, correct: {correct}, accuracy: {accuracy:.6f}, recall: {recall:.6f}, map50: {map50:.6f}, map50_95: {map50_95:.6f}".format(
            tested=i+1,
            correct=i+1,
            accuracy=i/total,
            recall=i/total,
            map50=i/total,
            map50_95=i/total
        ))


def user_validation():
    global model
    INPUTS_DIR = "/dataset/user_validation/images"
    OUTPUTS_DIR = "/dataset/user_validation/output"
    HP_CONFIDENCE = float(os.environ["HP_CONFIDENCE"])
    
    raise NotImplementedError()


def training_mode():
    if not prepare_dataset_3(): exit(1)
    train()
    val()


def user_validation_mode():
    user_validation()


def validation_mode():
    """
    running in validation mode.
    """
    if not prepare_dataset_3(ssplits=["test", "test", "test"], dsplits=["train", "val", "test"]): exit(1)
    val()

def print_environ():
    HP_EPOCHES = os.environ.get("HP_EPOCHES", "No HP_EPOCHES")
    HP_BATCH_SIZE = os.environ.get("HP_BATCH_SIZE", "No HP_BATCH_SIZE")
    HP_LEARNING_RATE = os.environ.get("HP_LEARNING_RATE", "No HP_LEARNING_RATE")
    HP_WEIGHT_DECAY = os.environ.get("HP_WEIGHT_DECAY", "No HP_WEIGHT_DECAY")
    HP_MOMENTUN = os.environ.get("HP_MOMENTUN", "No HP_MOMENTUN")
    HP_CONFIDENCE = os.environ.get("HP_CONFIDENCE", "No HP_CONFIDENCE")
    print(f"===on_start===")
    print(f"HP_EPOCHES: {HP_EPOCHES}")
    print(f"HP_BATCH_SIZE: {HP_BATCH_SIZE}")
    print(f"HP_LEARNING_RATE: {HP_LEARNING_RATE}")
    print(f"HP_WEIGHT_DECAY: {HP_WEIGHT_DECAY}")
    print(f"HP_MOMENTUN: {HP_MOMENTUN}")
    print(f"HP_CONFIDENCE: {HP_CONFIDENCE}")
    print(f"======")

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
