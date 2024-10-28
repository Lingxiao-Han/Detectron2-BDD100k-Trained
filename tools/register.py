
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

def register_bdd100k():
    train_image_path = "/Users/leo/Documents/Oin Auto Intern/detectron2/BDD100K/images/train"
    train_json_path = "/Users/leo/Documents/Oin Auto Intern/detectron2/BDD100K/labels/det_train.json"

    val_image_path = "/Users/leo/Documents/Oin Auto Intern/detectron2/BDD100K/images/val"
    val_json_path = "/Users/leo/Documents/Oin Auto Intern/detectron2/BDD100K/labels/det_val.json"


    register_coco_instances("bdd100k_train", {}, train_json_path, train_image_path)
    register_coco_instances("bdd100k_val", {}, val_json_path, val_image_path)

    # Define the class names (if not already included in the JSON)
    bdd_classes = [
        "bike", "bus", "car", "motor", "person",
        "rider", "traffic light", "traffic sign", "train", "truck"
    ]

    # Set metadata for the datasets
    MetadataCatalog.get("bdd100k_train").thing_classes = bdd_classes
    MetadataCatalog.get("bdd100k_val").thing_classes = bdd_classes
