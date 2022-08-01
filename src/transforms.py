import torch
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import json
import numpy as np
from src.type_validation.pytorch_type_validator import PytorchTypeValidator
from src.dataset.target import Target
from src.augmentation.transform_menu import IMGAUG_TRANSFORMATIONS
from typing import Union, Dict, List, Optional
from src.utils.normalizer import Normalizer
from src.type_validation.type_converter import TypeConverter


class ImgAugTransforms(object):
    """
    Inputs:
        cv_task (str): options = classification, detection, semseg
        config_file (path): path to augmentation config file
        bbox_keep_fraction (float): After augmentation, bboxes with a fraction of their area inside the augmented image
                    greater than the bbox_keep_fraction will be kept. Bboxes with a fraction inside the image less than
                    bbox_keep_fraction will be discarded
    """

    def __init__(
        self, cv_task: str, config_file: str = None, bbox_keep_fraction=0.75
    ) -> None:
        self.bbox_keep_fraction = bbox_keep_fraction

        assert cv_task.lower() in ["classification", "detection", "semseg"]
        self.cv_task = cv_task

        if config_file:
            self.aug = self.build_transform_from_config(config_file)
        else:
            self.aug = iaa.Sequential(
                [iaa.Sometimes(0.5, iaa.Fliplr()), iaa.Crop(percent=(0, 0.2))]
            )

    def transform_tuple(self, x: Union[tuple, list]) -> Union[tuple, list]:
        """
        in json, there is no tuple, but imgaug treats lists and tuples differently
        Per the documentation, a tuple creates a uniform distribution between x[0] and x[1]
        I think there is an issue with random generation in detectron2.
        Lists are treated as something to sample the value from, so we can just generate
        a distribution
        """
        if isinstance(x, list):
            if len(x) == 2:
                return tuple(x)
            else:
                return x
        else:
            return x

    def get_tf(self, td: dict) -> Union[np.ndarray, torch.Tensor]:
        prob = td["probability"]
        kwargs = {}
        for k, v in td.get("kwargs", {}).items():
            kwargs[k] = self.transform_tuple(v)
        return iaa.Sometimes(prob, IMGAUG_TRANSFORMATIONS[td["type"]](**kwargs))

    def build_transform_from_config(self, transform_config_json: str) -> iaa.Sequential:
        with open(transform_config_json, "r") as f:
            config = json.load(f)

        t_list = []
        for t in config["transformers"]:
            t_list.append(self.get_tf(t))

        seq = iaa.Sequential(t_list)
        return seq

    def _classification_augment(
        self, image: Union[np.ndarray, torch.Tensor], aug_det: iaa.Sequential
    ) -> torch.Tensor:

        image_aug = aug_det(image=image)
        return self._unprepare_after_aug(image_aug=image_aug)

    def _objdet_augment(
        self,
        image: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor],
        aug_det: iaa.Sequential,
    ) -> tuple:

        bboxes = [BoundingBox(x[0], x[1], x[2], x[3]) for x in target["boxes"]]
        bboxes = BoundingBoxesOnImage(bboxes, image.shape)
        image_aug, bboxes_aug = aug_det(image=image, bounding_boxes=bboxes)
        clipped_boxes = bboxes_aug.remove_out_of_image_fraction(self.bbox_keep_fraction)
        target["boxes"] = torch.from_numpy(clipped_boxes.to_xyxy_array())
        image_aug = self._unprepare_after_aug(image_aug=image_aug)
        return image_aug, target

    def _semseg_augment(
        self,
        image: Union[np.ndarray, torch.Tensor],
        segmap: Union[np.ndarray, torch.Tensor],
        aug_det: iaa.Sequential,
    ) -> tuple:

        image_aug, segmap = aug_det(image=image, segmentation_maps=segmap)
        image_aug = self._unprepare_after_aug(image_aug=image_aug)
        return image_aug, segmap

    def _prepare_for_aug(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        image = TypeConverter.coerce_channel_to_axis1(image)
        image = Normalizer.normalize_0_to_255(image)
        image = TypeConverter.coerce_to_torch(image)
        image = image.type(torch.uint8)
        return np.moveaxis(image.numpy(), 0, -1)

    def _unprepare_after_aug(
        self, image_aug: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        return torch.from_numpy(np.moveaxis(image_aug.copy(), -1, 0))

    def __call__(
        self,
        image: Union[np.ndarray, torch.Tensor],
        target: Union[Dict, Target] = None,
        segmap: Union[np.ndarray, torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple]:

        """
        Expects a 3 dimensional image with dim_0 = channels
        """

        # if not PytorchTypeValidator.image_is_correct_type(image):
        #     image = PytorchTypeValidator.coerce_image_to_correct_type(image)

        if isinstance(target, Target):
            target = target.to_retinanet_dict()
        assert target == None or type(target) == dict

        # TODO: validate segmap

        image = self._prepare_for_aug(image=image)
        aug_det = self.aug.to_deterministic()

        if self.cv_task.lower() == "classification":
            image_aug = self._classification_augment(image=image, aug_det=aug_det)
            return image_aug
        elif self.cv_task.lower() == "detection":
            assert type(target) == dict
            assert "boxes" and "labels" in target.keys()
            image_aug, target = self._objdet_augment(
                image=image, target=target, aug_det=aug_det
            )
            return image_aug, target
        elif self.cv_task.lower() == "semseg":
            segmap = SegmentationMapsOnImage(segmap, shape=image.shape)
            image_aug, segmap = self._semseg_augment(
                image=image, segmap=segmap, aug_det=aug_det
            )
            return image_aug, segmap
        else:
            raise Exception(
                "The type of augmentation neede could not be determined. Please ensure that target = None \
                or that target is a dictionary containing bboxes and labels"
            )
