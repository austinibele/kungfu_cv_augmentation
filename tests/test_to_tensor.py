import pytest
import torch
from src.augmentation.to_tensor import ToTensor


def test_classification():
    to_tensor = ToTensor(cv_task="classification")
    image = to_tensor(image=torch.ones([100, 200, 3]))
    assert torch.is_tensor(image)
    assert image.shape == (3, 100, 200)


def test_detection():
    to_tensor = ToTensor(cv_task="detection")
    image, target = to_tensor(image=torch.ones([100, 200, 3]), target={})
    assert torch.is_tensor(image)
    assert image.shape == (3, 100, 200)
    assert target == {}


def test_semseg():
    to_tensor = ToTensor(cv_task="semseg")
    image, segmap = to_tensor(
        image=torch.ones([100, 200, 3]), segmap=torch.zeros([100, 200, 3])
    )
    assert torch.is_tensor(image)
    assert image.shape == (3, 100, 200)
    assert torch.is_tensor(segmap)
    assert segmap.shape == (100, 200, 3)
