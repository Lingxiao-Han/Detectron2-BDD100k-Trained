from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper
from detectron2.data import transforms as T
from detectron2.structures import BoxMode
import torch

class CustomDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        self.resize_transform = T.Resize((416, 416))

    def __call__(self, dataset_dict):
        """
        Add custom resize logic on top of the default DatasetMapper.
        """
        dataset_dict = dataset_dict.copy() 
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)

        utils.check_image_size(dataset_dict, image)

        transform = self.resize_transform.get_transform(image)
        image = transform.apply_image(image)

        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1)) 

        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(obj, transform, image.shape[:2])
                for obj in dataset_dict["annotations"]
            ]
            instances = utils.annotations_to_instances(annos, image.shape[:2])
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
