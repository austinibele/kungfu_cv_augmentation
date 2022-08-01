import os
from src.augmentation.transforms import ImgAugTransforms
from typing import Union
from src.augmentation.compose import Compose
from src.augmentation.rescale import Rescale
from src.augmentation.to_tensor import ToTensor


def build_transform(
    cv_task: str,
    train: bool = True,
    imgaug_config: Union[str, os.PathLike] = None,
    rescale: bool = False,
) -> Compose:

    """
    This method composes transforms including a rescaling transform
    train (bool): Whether or not to use training transformations
    imgaug_confg (str): Location of file to use for image augmentation
    """

    assert cv_task.lower() in ["classification", "detection", "semseg"]
    transforms = []
    transforms.append(ToTensor(cv_task=cv_task))

    if train and imgaug_config:
        transforms.append(ImgAugTransforms(config_file=imgaug_config, cv_task=cv_task))
    else:
        transforms.append(
            ImgAugTransforms(
                config_file="config/augmentation/img_aug_none.json", cv_task=cv_task
            )
        )
    if rescale:
        transforms.append(Rescale(cv_task=cv_task))

    return Compose(transforms=transforms, cv_task=cv_task)
