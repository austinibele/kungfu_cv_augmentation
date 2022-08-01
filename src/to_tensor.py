import numpy as np
import torch
import torch.nn.functional as F
from typing import Union
from src.type_validation.pytorch_type_validator import PytorchTypeValidator


class ToTensor(object):
    def __init__(self, cv_task: str) -> None:
        self.cv_task = cv_task

    def _is_correct_type(self, image: Union[np.ndarray, torch.Tensor]) -> bool:
        if (
            torch.is_tensor(image)
            and image.shape[0] == 3
            and image.dtype == torch.float32
        ):
            return True
        else:
            return False

    def __call__(
        self,
        image: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor] = None,
        segmap: Union[np.ndarray, torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple]:

        if not PytorchTypeValidator.image_is_correct_type(image):
            image = PytorchTypeValidator.coerce_image_to_correct_type(image)
        if self.cv_task.lower() == "classification":
            return image
        elif self.cv_task.lower() == "detection":
            return image, target
        elif self.cv_task.lower() == "semseg":
            return image, segmap
        else:
            raise Exception(
                "There was an error and transformations were not applied. Please check \
                the inputs to the transform class attribute in the dataset __getitem__ method."
            )
