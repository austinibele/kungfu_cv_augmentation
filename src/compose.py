import numpy as np
import torch
from typing import Dict, Union, List
from src.augmentation.transforms import ImgAugTransforms
from src.augmentation.to_tensor import ToTensor


class Compose(object):
    def __init__(
        self, transforms: List[Union[ImgAugTransforms, ToTensor]], cv_task: str
    ) -> Union[Union[torch.tensor, np.ndarray], tuple]:
        self.transforms = transforms
        self.cv_task = cv_task

    def __call__(
        self,
        image: Union[np.ndarray, torch.Tensor],
        target: Dict = None,
        segmap: Union[np.ndarray, torch.Tensor] = None,
    ) -> Union[tuple, np.ndarray, torch.Tensor]:

        if self.cv_task.lower() == "classification":
            for t in self.transforms:
                image = t(image)
            return image
        elif self.cv_task.lower() == "detection":
            for t in self.transforms:
                image, target = t(image, target)
            return image, target
        elif self.cv_task.lower() == "semseg":
            for t in self.transforms:
                image, segmap = t(image, segmap)
            return image, segmap
        else:
            raise Exception(
                "There was an error and transformations were not applied. You defined the cv_task as \
                {self.cv_task}. Are you sure this was spelled correctly? Valid options include 1.'classification', 2.'detection', and 3.'semseg'"
            )
