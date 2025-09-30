# -*- coding: utf-8 -*-
"""
Author: Prudhvi Chekuri
Date: 2025-09-27
Version: 1.0
"""

# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
    """Applies AutoContrast to an image.
    
    Args:
        img (PIL.Image): The input image.
    
    Returns:
        PIL.Image: The image with autocontrast applied.
    """
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    """Adjusts the brightness of an image.
    
    Args:
        img (PIL.Image): The input image.
        v (float): The magnitude of the brightness adjustment.
        max_v (float): The maximum possible magnitude.
        bias (float, optional): A bias to add to the adjustment. Defaults to 0.
        
    Returns:
        PIL.Image: The brightness-adjusted image.
    """
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    """Adjusts the color balance of an image.
    
    Args:
        img (PIL.Image): The input image.
        v (float): The magnitude of the color adjustment.
        max_v (float): The maximum possible magnitude.
        bias (float, optional): A bias to add to the adjustment. Defaults to 0.
        
    Returns:
        PIL.Image: The color-adjusted image.
    """
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    """Adjusts the contrast of an image.
    
    Args:
        img (PIL.Image): The input image.
        v (float): The magnitude of the contrast adjustment.
        max_v (float): The maximum possible magnitude.
        bias (float, optional): A bias to add to the adjustment. Defaults to 0.
        
    Returns:
        PIL.Image: The contrast-adjusted image.
    """
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    """Applies Cutout to an image, obscuring a random square region.
    
    The size of the square is relative to the image size.
    
    Args:
        img (PIL.Image): The input image.
        v (float): The magnitude of the cutout size.
        max_v (float): The maximum possible magnitude.
        bias (float, optional): A bias to add to the adjustment. Defaults to 0.
        
    Returns:
        PIL.Image: The image with the cutout region.
    """
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    """Applies Cutout with an absolute pixel value.
    
    Args:
        img (PIL.Image): The input image.
        v (int): The side length of the square cutout region in pixels.
        
    Returns:
        PIL.Image: The image with the cutout region.
    """
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    
    if img.mode == 'L':  # Handle datasets that has gray scale images ex: MNIST
        color = 127
    else:
        color = (127, 127, 127)

    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    """Applies histogram equalization to an image.
    
    Args:
        img (PIL.Image): The input image.
        
    Returns:
        PIL.Image: The equalized image.
    """
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    """Returns the original image without any changes.
    
    Args:
        img (PIL.Image): The input image.
        
    Returns:
        PIL.Image: The unchanged input image.
    """
    return img


def Invert(img, **kwarg):
    """Inverts the colors of an image.
    
    Args:
        img (PIL.Image): The input image.
        
    Returns:
        PIL.Image: The inverted image.
    """
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    """Reduces the number of bits for each color channel.
    
    Args:
        img (PIL.Image): The input image.
        v (int): The number of bits to keep for each channel.
        max_v (int): The maximum possible value for v.
        bias (int, optional): A bias to add to the parameter. Defaults to 0.
        
    Returns:
        PIL.Image: The posterized image.
    """
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    """Rotates an image by a random angle.
    
    Args:
        img (PIL.Image): The input image.
        v (int): The magnitude of the rotation angle.
        max_v (int): The maximum possible angle.
        bias (int, optional): A bias to add to the parameter. Defaults to 0.
        
    Returns:
        PIL.Image: The rotated image.
    """
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    """Adjusts the sharpness of an image.
    
    Args:
        img (PIL.Image): The input image.
        v (float): The magnitude of the sharpness adjustment.
        max_v (float): The maximum possible magnitude.
        bias (float, optional): A bias to add to the adjustment. Defaults to 0.
        
    Returns:
        PIL.Image: The sharpness-adjusted image.
    """
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    """Shears an image along the X-axis.
    
    Args:
        img (PIL.Image): The input image.
        v (float): The magnitude of the shear.
        max_v (float): The maximum possible magnitude.
        bias (float, optional): A bias to add to the parameter. Defaults to 0.
        
    Returns:
        PIL.Image: The sheared image.
    """
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    """Shears an image along the Y-axis.
    
    Args:
        img (PIL.Image): The input image.
        v (float): The magnitude of the shear.
        max_v (float): The maximum possible magnitude.
        bias (float, optional): A bias to add to the parameter. Defaults to 0.
        
    Returns:
        PIL.Image: The sheared image.
    """
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    """Inverts pixel values above a certain threshold.
    
    Args:
        img (PIL.Image): The input image.
        v (int): The solarization threshold.
        max_v (int): The maximum possible threshold.
        bias (int, optional): A bias to add to the parameter. Defaults to 0.
        
    Returns:
        PIL.Image: The solarized image.
    """
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    """Adds a value to pixels and then solarizes.
    
    Args:
        img (PIL.Image): The input image.
        v (int): The value to add to each pixel.
        max_v (int): The maximum possible value for v.
        bias (int, optional): A bias to add to the parameter. Defaults to 0.
        threshold (int, optional): The solarization threshold. Defaults to 128.
        
    Returns:
        PIL.Image: The transformed image.
    """
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    """Translates an image along the X-axis.
    
    Args:
        img (PIL.Image): The input image.
        v (float): The magnitude of the translation.
        max_v (float): The maximum possible magnitude.
        bias (float, optional): A bias to add to the parameter. Defaults to 0.
        
    Returns:
        PIL.Image: The translated image.
    """
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    """Translates an image along the Y-axis.
    
    Args:
        img (PIL.Image): The input image.
        v (float): The magnitude of the translation.
        max_v (float): The maximum possible magnitude.
        bias (float, optional): A bias to add to the parameter. Defaults to 0.
        
    Returns:
        PIL.Image: The translated image.
    """
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    """Scales a float parameter.
    
    Args:
        v (float): The input value.
        max_v (float): The maximum value for the parameter.
        
    Returns:
        float: The scaled parameter.
    """
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    """Scales an integer parameter.
    
    Args:
        v (int): The input value.
        max_v (int): The maximum value for the parameter.
        
    Returns:
        int: The scaled parameter.
    """
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    """Returns the augmentation pool used in the FixMatch paper.
    
    Returns:
        list: A list of tuples, where each tuple contains an augmentation
              function, its maximum value, and a bias.
    """
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs


def my_augment_pool():
    """Returns a custom augmentation pool for testing.
    
    Returns:
        list: A list of tuples with augmentation functions and their parameters.
    """
    augs = [(AutoContrast, None, None),
            (Brightness, 1.8, 0.1),
            (Color, 1.8, 0.1),
            (Contrast, 1.8, 0.1),
            (Cutout, 0.2, 0),
            (Equalize, None, None),
            (Invert, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 1.8, 0.1),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (SolarizeAdd, 110, 0),
            (TranslateX, 0.45, 0),
            (TranslateY, 0.45, 0)]
    return augs


class RandAugmentPC(object):
    """A RandAugment class that uses a custom augmentation pool."""
    def __init__(self, n, m):
        """Initializes the RandAugmentPC object.
        
        Args:
            n (int): The number of augmentation operations to apply.
            m (int): The magnitude of the augmentations (1 to 10).
        """
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = my_augment_pool()

    def __call__(self, img):
        """Applies a sequence of random augmentations to an image.
        
        Args:
            img (PIL.Image): The input image.
            
        Returns:
            PIL.Image: The augmented image.
        """
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img


class RandAugmentMC(object):
    """A RandAugment class that uses the FixMatch augmentation pool."""
    def __init__(self, n, m):
        """Initializes the RandAugmentMC object.
        
        Args:
            n (int): The number of augmentation operations to apply.
            m (int): The magnitude for augmentations (1 to 10).
        """
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        """Applies a sequence of random augmentations to an image.
        
        Args:
            img (PIL.Image): The input image.
            
        Returns:
            PIL.Image: The augmented image.
        """
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img