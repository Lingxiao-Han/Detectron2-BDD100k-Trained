import json
import os

bdd_json_path = "/root/Object-Detection/BDD100K/labels/det_train.json"
coco_json_path = "/root/Object-Detection/BDD100K/labels/det_train_coco.json"

categories = [
    {"id": 1, "name": "bike"},
    {"id": 2, "name": "bus"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "motor"},
    {"id": 5, "name": "person"},
    {"id": 6, "name": "rider"},
    {"id": 7, "name": "traffic light"},
    {"id": 8, "name": "traffic sign"},
    {"id": 9, "name": "train"},
    {"id": 10, "name": "truck"}
]

def convert_bdd_to_coco(bdd_data):
    coco_format = {"images": [], "annotations": [], "categories": categories}
    annotation_id = 0

    for img_id, item in enumerate(bdd_data):
        coco_format["images"].append({
            "id": img_id,
            "file_name": item["name"],
            "width": 1280,  
            "height": 720   
        })

        labels = item.get("labels", [])
        for label in labels:
            if "box2d" in label:
                x1 = label["box2d"]["x1"]
                y1 = label["box2d"]["y1"]
                x2 = label["box2d"]["x2"]
                y2 = label["box2d"]["y2"]

                bbox = [x1, y1, x2 - x1, y2 - y1]

                # 查找类别 ID
                category_id = next(
                    (cat["id"] for cat in categories if cat["name"] == label["category"]), None
                )
                if category_id is None:
                    continue 

                # 添加 annotation
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                })
                annotation_id += 1

    return coco_format

with open(bdd_json_path, "r") as f:
    bdd_data = json.load(f)

coco_data = convert_bdd_to_coco(bdd_data)

with open(coco_json_path, "w") as f:
    json.dump(coco_data, f)

print(f"转换完成，已保存为 {coco_json_path}")
