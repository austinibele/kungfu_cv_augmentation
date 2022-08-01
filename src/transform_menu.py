import imgaug.augmenters as iaa
"""
These transformations come from imgaug.augmenters
"""
IMGAUG_TRANSFORMATIONS = {
    "sharpen": iaa.Sharpen,
    "emboss": iaa.Emboss,
    "gaussian_blur": iaa.GaussianBlur,
    "perspective_transform": iaa.PerspectiveTransform,
    "affine": iaa.Affine,
    "color_temp": iaa.ChangeColorTemperature,
    "salt_and_pepper": iaa.SaltAndPepper,
    "flip_lr": iaa.Fliplr,
    "flip_ud": iaa.Flipud,
    "contrast": iaa.LinearContrast,
    "multiply": iaa.Multiply,
    "resize": iaa.Resize,
    "crop": iaa.Crop,
    "dropout": iaa.Dropout,
    "add": iaa.Add,
    "rot90": iaa.geometric.Rot90
}
