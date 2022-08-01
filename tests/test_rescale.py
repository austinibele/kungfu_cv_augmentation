import torch
import pytest
from src.augmentation.rescale import Rescale


def test_classification():
    rescale = Rescale(cv_task="classification")
    out = rescale(image=torch.ones([200, 200, 3]))
    assert out != None
    assert torch.is_tensor(out)


def test_detection():
    rescale = Rescale(cv_task="detection")
    out = rescale(image=torch.ones([200, 200, 3]) * 255, target={})
    assert torch.is_tensor(out[0])
    assert torch.max(out[0]) == 1
    assert isinstance(out[1], dict)


def test_semseg():
    rescale = Rescale(cv_task="semseg")
    out = rescale(
        image=torch.ones([200, 200, 3]) * 255, segmap=torch.zeros([200, 200, 3])
    )
    assert torch.is_tensor(out[0])
    assert torch.max(torch.max(out[0])) == 1
    assert torch.is_tensor(out[1])
