import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

# Step 1: Load the config
cfg = get_cfg()
cfg.merge_from_file("/root/Object-Detection/tools/output/bdd100k_faster_rcnn/config.yaml")  # Path to your config
cfg.MODEL.WEIGHTS = "/root/Object-Detection/tools/output/bdd100k_faster_rcnn/model_final.pth"  # Path to your trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a custom threshold for predictions
cfg.MODEL.DEVICE = "cuda"  # Use GPU if available, else "cpu"

# Step 2: Create a predictor
predictor = DefaultPredictor(cfg)

# Step 3: Load an image
image_path = "/root/Object-Detection/blog-image-signal.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Step 4: Run inference
outputs = predictor(image)

# Step 5: Visualize predictions
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Save or display the result
result_image = v.get_image()[:, :, ::-1]
cv2.imwrite("/root/Object-Detection/blog-image-signal_OUT.jpg", result_image)  # Save the result
cv2.imshow("Predictions", result_image)
cv2.waitKey(0)
