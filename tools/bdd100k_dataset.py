# bdd100k_dataset.py
import sys
import logging
import os
from collections import OrderedDict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog


def register_bdd100k():
    train_image_path = "/root/Object-Detection/BDD100K/images/train"
    train_json_path = "/root/Object-Detection/BDD100K/labels/det_train_coco.json"

    val_image_path = "/root/Object-Detection/BDD100K/images/val"
    val_json_path = "/root/Object-Detection/BDD100K/labels/det_val_coco.json"

    register_coco_instances("bdd100k_train", {}, train_json_path, train_image_path)
    register_coco_instances("bdd100k_val", {}, val_json_path, val_image_path)

    bdd_classes = [
        "bike", "bus", "car", "motor", "person",
        "rider", "traffic light", "traffic sign", "train", "truck"
    ]

    MetadataCatalog.get("bdd100k_train").thing_classes = bdd_classes
    MetadataCatalog.get("bdd100k_val").thing_classes = bdd_classes
