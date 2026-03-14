import mlx.core as mx
import numpy as np
import random
from PIL import Image
from config import Config
# ImageNet 統計値
from utils.normalize import ImageNet as nm
from utils.preprocess import Preprocess as Loader

import torch
import torch.utils.data as data
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as F
from torch import optim
import torch.utils.data as data
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import VisionDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import (
    Compose,
    RandomResizedCrop,
    RandomCrop,
    RandomHorizontalFlip,
    ColorJitter,
    GaussianBlur,
    Resize,
    ToTensor,
    Normalize,
    Lambda,
    InterpolationMode,
    RandomErasing,#追加
    RandomRotation,
)
from dataclasses import dataclass
import random

from torchvision.transforms.v2 import (
    AutoAugment,
    AutoAugmentPolicy
)
from tqdm import tqdm

#
class AugmentationParams:
    """
    AugmentationParams(
        hflip=hlip, # 水平反転の確率。
        degrees=degrees,# ランダム回転の最大角度。
        jitter_param={
            "brightness":[0.1-0.5],
            "contrast":[0.1-0.5],
            "saturation":[0.1-0.5],
            "hue":[0.05-0.2]
        },
        size=(512, 512)
    )
    """
    def __init__(
        self,
        degrees:int=10, # Rotate
        hflip:float=0.5,
        jitter_param:dict=None,
        size:tuple=(224, 224),# Crop Size
        crop:bool=False,
        crop_size:tuple=(224, 224),
        crop_scale:tuple=(0.7, 1.0),
        crop_ratio:tuple=(1.0, 1.0),
        cut_out=None
    ):
        self.degrees = degrees
        self.hflip = hflip
        self.jitter_param = jitter_param
        if jitter_param is None:
            self.jitter_param = {"brightness":0.0, "contrast":0.0, "saturation":0.0, "hue":0.0}
        self.crop = crop
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        # ColorJitter
        self.jitter=None if jitter_param is None else ColorJitter(**self.jitter_param)

# データ拡張：実体クラス
class Augmentation:
    def __init__(self, parameter):
        """
        この処理はTensor化/正規化前に行う。
        """
        self.parameter = parameter
        self.setting()

    def setting(self):
        self.hflip = self.parameter.hflip
        self.degrees = self.parameter.degrees
        self.jitter_param = self.parameter.jitter_param
        self.crop = self.parameter.crop
        self.crop_size = self.parameter.crop_size
        self.crop_scale = self.parameter.crop_scale
        self.crop_ratio = self.parameter.crop_ratio

        # ColorJitter
        self.jitter=ColorJitter(**self.parameter.jitter_param)

    def __call__(self, image, depth, target):
        # RandomResizeCrop
        if self.crop is True:
            i, j, h, w = RandomResizedCrop.get_params(image, scale=self.crop_scale, ratio=self.crop_ratio)
            image = F.resized_crop(image, i, j, h, w, self.crop_size, interpolation=InterpolationMode.BILINEAR)
            depth = F.resized_crop(depth, i, j, h, w, self.crop_size, interpolation=InterpolationMode.BILINEAR)
            if target is not None:
                target = F.resized_crop(target, i, j, h, w, self.crop_size, interpolation=InterpolationMode.NEAREST)

        # Flip
        if random.random() < self.hflip:
            image = F.hflip(image)
            depth = F.hflip(depth)
            if target is not None:
                target = F.hflip(target)

        # Rotate
        if self.degrees > 0:
            angle = RandomRotation.get_params([-self.degrees, self.degrees])
            image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            depth = F.rotate(depth, angle, interpolation=InterpolationMode.NEAREST)
            if target is not None:
                target = F.rotate(target, angle, interpolation=InterpolationMode.NEAREST)

        # jitter
        image = self.jitter(image)
        return image, depth, target
