import pytest
from src.augmentation.compose import Compose
import numpy as np
import torch


def test_classification():
    compose = Compose(transforms=[], cv_task="classification")
    out = compose(image=torch.ones([300, 300, 3]))
    assert torch.is_tensor(out)
    assert torch.isclose(out.sum(), torch.ones([300, 300, 3]).sum())


def test_detections():
    compose = Compose(transforms=[], cv_task="detection")
    out = compose(image=torch.ones([300, 300, 3]), target={})
    assert torch.is_tensor(out[0])
    assert isinstance(out[1], dict)


def test_semseg():
    compose = Compose(transforms=[], cv_task="semseg")
    out = compose(image=torch.ones([300, 300, 3]), segmap=np.ones([300, 300, 3]))
    assert torch.is_tensor(out[0])
    assert isinstance(out[1], np.ndarray)
