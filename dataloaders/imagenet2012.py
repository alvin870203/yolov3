"""
Training & validation dataloaders of ImageNet2012 classification dataset.
"""

from dataclasses import dataclass
import warnings
from typing import Any, Callable, cast, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F, InterpolationMode
from torchvision.transforms.v2.functional._utils import _FillType


class Resize(v2.Resize):
    def __init__(
            self,
            letterbox: bool,
            fill:  Union[_FillType, Dict[Union[Type, str], _FillType]] = 0,
            **kwargs,  # params for v2.Resize
        ) -> None:
        super().__init__(**kwargs)
        self.size = self.size + self.size if len(self.size) == 1 else self.size
        self.letterbox = letterbox
        self.fill = fill
        self._fill = v2._utils._setup_fill_arg(fill)
        self.padding_mode = 'constant'  # only support constant padding mode for bounding boxes

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_h, orig_w = v2._utils.query_size(flat_inputs)
        new_h, new_w = self.size
        if not self.letterbox:
            return dict(size=(new_h, new_w))
        else:  # do letterbox
            r_h, r_w = new_h / orig_h, new_w / orig_w
            r = min(r_h, r_w)
            new_unpad_h, new_unpad_w = round(orig_h * r), round(orig_w * r)
            pad_left = pad_right = pad_top = pad_bottom = 0
            if r_w < r_h:
                diff = new_h - new_unpad_h
                pad_top += (diff // 2)
                pad_bottom += (diff - pad_top)
            else:  # r_h <= r_w:
                diff = new_w - new_unpad_w
                pad_left += (diff // 2)
                pad_right += (diff - pad_left)
            padding = [pad_left, pad_top, pad_right, pad_bottom]
            return dict(size=(new_unpad_h, new_unpad_w), padding=padding)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = self._call_kernel(F.resize, inpt, size=params['size'],
                                 interpolation=self.interpolation, antialias=self.antialias)
        if self.letterbox:
            fill = v2._utils._get_fill(self._fill, type(inpt))
            inpt = self._call_kernel(F.pad, inpt, padding=params["padding"], fill=fill, padding_mode=self.padding_mode)
        return inpt


@dataclass
class ImageNetConfig:
    img_h: int = 256
    img_w: int = 256
    imgs_mean: Tuple = (0.485, 0.456, 0.406)
    imgs_std: Tuple = (0.229, 0.224, 0.225)
    aug_type: str = 'simple-aug'  # 'simple-aug', 'tv-aug', or 'voc-aug'
    scale_min: float = 0.08  # 0.08 for random resized crop, 0.5 for affine transform
    scale_max: float = 1.0  # 1.0 for random resized crop, 1.5 for affine transform
    ratio_min: float = 3.0 / 4.0  # 3.0 / 4.0 for random resized crop, 0.5 for affine transform
    ratio_max: float = 4.0 / 3.0  # 4.0 / 3.0 for random resized crop, 2.0 for affine transform
    perspective: float = 0.1
    degrees: float = 0.5
    translate: float = 0.25
    shear: float = 0.5
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.7
    hue: float = 0.015
    flip_p: float = 0.5
    letterbox: float = True
    fill: Tuple = (123.0, 117.0, 104.0)


class ImageNetTrainDataLoader(DataLoader):
    def __init__(self, config: ImageNetConfig, data_dir, batch_size, num_workers, shuffle=True, pin_memory=True,
                 nano=False):  # if True: use only the first two images as the entire dataset
        self.config = config
        if config.aug_type == 'simple-aug':
            transform = v2.Compose([
                v2.ToImage(),
                v2.ColorJitter(brightness=config.brightness, contrast=config.contrast,
                               saturation=config.saturation, hue=config.hue),
                v2.RandomResizedCrop(size=(config.img_h, config.img_w), scale=(config.scale_min, config.scale_max),
                                     ratio=(config.ratio_min, config.ratio_max), antialias=True),
                v2.RandomHorizontalFlip(p=config.flip_p),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
            ])
        elif config.aug_type == 'tv-aug':
            transform = v2.Compose([
                v2.ToImage(),
                v2.RandomResizedCrop(size=(config.img_h, config.img_w), scale=(config.scale_min, config.scale_max),
                                     ratio=(config.ratio_min, config.ratio_max), antialias=True),
                v2.RandomHorizontalFlip(p=config.flip_p),
                v2.TrivialAugmentWide(fill=config.fill),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
            ])
        elif config.aug_type == 'voc-aug':
            transform = v2.Compose([
                v2.ColorJitter(brightness=config.brightness, contrast=config.contrast,
                               saturation=config.saturation, hue=config.hue),
                # perspective & affine transform on PIL image solves the issue with black borders
                v2.RandomPerspective(distortion_scale=config.perspective, fill=config.fill,
                                     interpolation=InterpolationMode.BICUBIC),
                v2.RandomAffine(degrees=config.degrees, translate=(config.translate, config.translate),
                                scale=(config.scale_min, config.scale_max),
                                shear=(-config.shear, config.shear, -config.shear, config.shear), fill=config.fill,
                                interpolation=InterpolationMode.BICUBIC),
                Resize(size=(config.img_h, config.img_w), letterbox=config.letterbox, fill=config.fill, antialias=True,
                       interpolation=InterpolationMode.BICUBIC),
                v2.ToImage(),  # bicubic interpolation is not supported for tensors
                v2.RandomHorizontalFlip(p=config.flip_p),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
            ])
        else:
            raise ValueError(f"Invalid augmentation type: {config.aug_type}")

        dataset = torchvision.datasets.ImageNet(data_dir, split='train', transform=transform)
        if nano:
            dataset = Subset(dataset, indices=range(2))
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=pin_memory)


class ImageNetValDataLoader(DataLoader):
    # Default shuffle=True since only eval partial data
    def __init__(self, config: ImageNetConfig, data_dir, batch_size, num_workers, shuffle=True, pin_memory=True,
                 nano=False):  # if True: use only the first two images as the entire dataset
        self.config = config
        if config.aug_type == 'simple-aug' or config.aug_type == 'tv-aug':
            transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(size=(int(config.img_h * 256 / 224), int(config.img_w * 256 / 224)), antialias=True),
                v2.CenterCrop(size=(config.img_h, config.img_w)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
            ])
        elif config.aug_type == 'voc-aug':
            transform = v2.Compose([
                v2.ToImage(),
                Resize(size=(config.img_h, config.img_w), letterbox=config.letterbox, fill=config.fill, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
            ])
        else:
            raise ValueError(f"Invalid augmentation type: {config.aug_type}")

        dataset = torchvision.datasets.ImageNet(data_dir, split='val', transform=transform)
        if nano:
            dataset = Subset(dataset, indices=range(2))
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=pin_memory)


@dataclass
class BlankImageNetConfig:
    img_h: int = 256
    img_w: int = 256
    imgs_mean: Tuple = (0.485, 0.456, 0.406)
    imgs_std: Tuple = (0.229, 0.224, 0.225)
    fill: Tuple = (123.0, 117.0, 104.0)


class BlankImageNetTrainDataLoader(DataLoader):
    """All images are set to zeros. Used for setting input-independent baseline."""
    def __init__(self, config: BlankImageNetConfig, data_dir, batch_size, num_workers, shuffle=True, pin_memory=True,
                 nano=False):  # if True: use only the first two images as the entire dataset
        self.config = config
        dataset = torchvision.datasets.ImageNet(
            data_dir, split='train',
            transform=v2.Compose([
                v2.ToImage(),
                v2.Lambda(lambda inp: tv_tensors.wrap(torch.tensor(config.fill,
                                                                   dtype=inp.dtype,
                                                                   device=inp.device).view(3, 1, 1).expand(inp.shape),
                                                      like=inp)
                          if isinstance(inp, tv_tensors.Image) else inp),
                v2.Resize(size=(config.img_h, config.img_w), interpolation=InterpolationMode.NEAREST, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
            ])
        )
        if nano:
            dataset = Subset(dataset, indices=range(2))
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=pin_memory)


class BlankImageNetValDataLoader(DataLoader):
    """All images are set to zeros. Used for setting input-independent baseline."""
    # Default shuffle=True since only eval partial data
    def __init__(self, config: BlankImageNetConfig, data_dir, batch_size, num_workers, shuffle=True, pin_memory=True,
                 nano=False):  # if True: use only the first two images as the entire dataset
        self.config = config
        dataset = torchvision.datasets.ImageNet(
            data_dir, split='val',
            transform=v2.Compose([
                v2.ToImage(),
                v2.Lambda(lambda inp: tv_tensors.wrap(torch.tensor(config.fill,
                                                                   dtype=inp.dtype,
                                                                   device=inp.device).view(3, 1, 1).expand(inp.shape),
                                                      like=inp)
                          if isinstance(inp, tv_tensors.Image) else inp),
                v2.Resize(size=(config.img_h, config.img_w), interpolation=InterpolationMode.NEAREST, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
            ])
        )
        if nano:
            dataset = Subset(dataset, indices=range(2))
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=pin_memory)


if __name__ == '__main__':
    # Test the dataloaders by `python -m dataloaders.imagenet2012` from the workspace directory
    import matplotlib.pyplot as plt
    data_dir = 'data/imagenet2012'
    config = ImageNetConfig()
    dataloader_train = ImageNetTrainDataLoader(config, data_dir, batch_size=32, num_workers=4)
    dataloader_val = ImageNetValDataLoader(config, data_dir, batch_size=32, shuffle=False, num_workers=4)
    print(f"{len(dataloader_train)=}")
    print(f"{len(dataloader_val)=}")
    example_imgs, example_labels = next(iter(dataloader_train))
    print(f"{example_imgs.shape=}; {example_labels.shape=}")
    # Unnormalize the image for plotting
    example_img = example_imgs[0]
    example_img = example_img * torch.tensor(config.imgs_std).reshape(3, 1, 1) + torch.tensor(config.imgs_mean).reshape(3, 1, 1)
    print(f"{example_img.shape=}")
    print(f"{example_img.mean()=}, {example_img.std()=}")
    print(f"{example_img.min()=}, {example_img.max()=}")
    # plt.imshow(example_img.permute(1, 2, 0))
    # plt.show()
