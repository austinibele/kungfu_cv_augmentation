import pytest
from src.augmentation.build_transforms import build_transform
from src.augmentation.compose import Compose


def test_build_transform():
    cv_task = "detection"
    imgaug_config = "config/augmentation/img_aug_example.json"
    out = build_transform(cv_task=cv_task, imgaug_config=imgaug_config)
    assert isinstance(out, Compose)
