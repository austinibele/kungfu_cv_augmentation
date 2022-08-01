import glob
import numpy as np
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from src.augmentation.transforms import ImgAugTransforms
from src.dataset.target import Target
from src.type_validation.type_converter import TypeConverter
from src.utils.normalizer import Normalizer
from PIL import Image

config_path = "tests/augmentation/test_config.json"
test_image_path = "tests/augmentation/test_image.jpeg"


def return_segmap(image):
    segmap = np.zeros((128, 128, 1), dtype=np.int32)
    segmap[28:71, 35:85, 0] = 1
    segmap[10:25, 30:45, 0] = 2
    segmap[10:25, 70:85, 0] = 3
    segmap[10:110, 5:10, 0] = 4
    segmap[118:123, 10:110, 0] = 5
    return segmap


def bbox():
    bbox = [290, 115, 405, 385]
    bbox = torch.tensor(bbox, dtype=torch.int).unsqueeze(dim=0)
    return bbox


def test_init():
    image_aug_object = ImgAugTransforms(cv_task="detection", config_file=config_path)


def test_load_image():
    img = read_image(test_image_path)


def test_draw_bbox():
    img = read_image(test_image_path)
    img = draw_bounding_boxes(image=img, boxes=bbox(), width=3)
    img = torchvision.transforms.ToPILImage()(img)
    img.show()


def test_img_aug_classification():
    image_aug_object = ImgAugTransforms(
        cv_task="classification", config_file=config_path
    )
    img = read_image(test_image_path)
    image_aug = image_aug_object(image=img)
    image_aug = image_aug.to(torch.uint8)
    image_aug = torchvision.transforms.ToPILImage()(image_aug)
    image_aug.show()


def test_img_aug_objdet():
    image_aug_object = ImgAugTransforms(cv_task="detection", config_file=config_path)
    img = read_image(test_image_path)
    target = {}
    target["boxes"] = bbox()
    target["labels"] = torch.FloatTensor(np.ones(1))
    image_aug, target = image_aug_object(image=img, target=target)
    image_aug = image_aug.to(torch.uint8)
    image_aug = draw_bounding_boxes(image=image_aug, boxes=target["boxes"], width=3)
    image_aug = torchvision.transforms.ToPILImage()(image_aug)
    image_aug.show()


def test_img_aug_objdet_from_target():
    image_aug_object = ImgAugTransforms(cv_task="detection", config_file=config_path)
    img = read_image(test_image_path)
    target = {}
    target["boxes"] = bbox()
    target["labels"] = torch.FloatTensor(np.ones(1))
    target = Target.from_retinanet_dict(target)
    image_aug, target = image_aug_object(image=img, target=target)
    image_aug = image_aug.to(torch.uint8)
    image_aug = draw_bounding_boxes(image=image_aug, boxes=target["boxes"], width=3)
    image_aug = torchvision.transforms.ToPILImage()(image_aug)
    image_aug.show()


def test_draw_semseg_map():
    img = read_image(test_image_path)
    img = np.moveaxis(img.numpy(), 0, -1)
    segmap = return_segmap(image=img)
    segmap = SegmentationMapsOnImage(segmap, shape=img.shape)
    img = segmap.draw_on_image(image=img)[0]
    img = torch.from_numpy(np.moveaxis(img, -1, 0))
    img = torchvision.transforms.ToPILImage()(img)
    img.show()


def test_img_aug_segmentation():
    image_aug_object = ImgAugTransforms(cv_task="semseg", config_file=config_path)
    img = read_image(test_image_path)
    segmap = return_segmap(image=img)
    image_aug, segmap = image_aug_object(image=img, segmap=segmap)
    image_aug = np.moveaxis(image_aug.numpy(), 0, -1)
    image_aug = image_aug.astype(np.uint8)
    image_aug = segmap.draw_on_image(image=image_aug)[0]
    image_aug = torch.from_numpy(np.moveaxis(image_aug, -1, 0))
    image_aug = torchvision.transforms.ToPILImage()(image_aug)
    image_aug.show()
