import numpy as np
import torch
from typing import Union


class Rescale:
    """
    This is needed to scale images between zero and one
    """

    def __init__(self, cv_task):
        self.cv_task = cv_task

    def __call__(
        self,
        image: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor] = None,
        segmap: Union[np.ndarray, torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple]:

        rescaled = image / (255.0)
        if self.cv_task.lower() == "classification":
            return rescaled.type(torch.FloatTensor)
        elif self.cv_task.lower() == "detection":
            return rescaled.type(torch.FloatTensor), target
        elif self.cv_task.lower() == "semseg":
            return rescaled.type(torch.FloatTensor), segmap
        else:
            raise Exception(
                "There was an error and transformations were not applied. Please check \
                the inputs to the transform class attribute in the dataset __getitem__ method."
            )
