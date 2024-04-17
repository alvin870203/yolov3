"""
Training & validation dataloaders of VOC detection dataset.
"""

from dataclasses import dataclass
import warnings
from typing import Any, Callable, cast, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F, InterpolationMode
from torchvision.transforms.v2.functional._utils import _FillType
from torchvision.datasets import wrap_dataset_for_transforms_v2


def add_img_border_box(x, y):
    """
    Add a bounding box around the image, to record the image border transform.
    Args:
        x (PIL.Image.Image): size(H, W), RGB, 0~255
        y (Dict): VOC annotation
            y['boxes'] (tv_tensors.BoundingBoxes): size(n_obj, 4), in pixels, XYXY format
            y['labels'] (Tensor): size(n_obj,), torch.int64, 1~n_class, with background class 0
    Returns:
        x (PIL.Image.Image): same as input
        y (Dict): VOC annotation
            y['boxes'] (tv_tensors.BoundingBoxes): size(n_obj+1, 4), with the last box as the border box
            y['labels'] (Tensor): size(n_obj+1,), with the last label as 0 (background class)
    """
    img_w, img_h = x.size
    border_box = torch.tensor([0, 0, img_w-1, img_h-1]).view(1, 4)
    y['boxes'] = tv_tensors.wrap(torch.cat((y['boxes'], border_box), dim=0), like=y['boxes'])
    y['labels'] = torch.cat((y['labels'], torch.tensor([0], dtype=torch.int64)))
    return x, y


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


class Voc2Yolov3(nn.Module):
    """
    Args:
        x (Tensor): size(3, img_h, img_w), RGB, 0~255
        y_voc (Dict): VOC annotation
            y_voc['boxes'] (tv_tensors.BoundingBoxes): size(n_obj+1, 4), in pixels, XYXY format
                with the last box as the border box
            y_voc['labels'] (Tensor): size(n_obj+1,), torch.int64, 1~n_class, with background class 0
                with the last label as 0 (background class) for border box
    Returns:
        x (Tensor): size(3, img_h, img_w), RGB, 0~255
        y_yolov3 (Tensor): size(n_obj, 6), torch.float32
            y_yolov3[i, 0] is the idx of the image in the batch, 0~batch_size-1, init with -1, assigned in collate_fn
            y_yolov3[i, 1] is the class index for the i-th object box, 0.0~float(n_class-1), no background class
            y_yolov3[i, 2:6] is the box coordinates for the i-th object box, normalized by img wh, CXCYWH format
        border (Tensor): size(4,), torch.float32, the border box normalized by img wh, XYXY format
    """
    def forward(self, x, y_voc):
        img_h, img_w = x.shape[-2:]
        cxcywh = box_convert(y_voc['boxes'][:-1], in_fmt='xyxy', out_fmt='cxcywh')  # :-1 to exclude border box
        cxcywhn = cxcywh / torch.tensor([img_w, img_h, img_w, img_h])
        n_obj = cxcywh.shape[0]
        y_yolov3 = torch.cat((
            torch.full((n_obj, 1), -1), (y_voc['labels'][:-1] - 1).unsqueeze(1), cxcywhn),  # - 1 to remove background class
            dim=1
        ).to(torch.float32)
        border = y_voc['boxes'][-1] / torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return x, y_yolov3, border

    @classmethod
    def inv_target_transform(self, x, y_yolov3):
        img_h, img_w = x.shape[-2:]
        cxcywh = y_yolov3[:, 2:6] * torch.tensor([img_w, img_h, img_w, img_h])
        xyxy = box_convert(cxcywh, in_fmt='cxcywh', out_fmt='xyxy')
        labels = y_yolov3[:, 1].to(torch.int64)
        y_voc = {'boxes': xyxy, 'labels': labels}
        return y_voc


@dataclass
class VocConfig:
    img_h: int = 416
    img_w: int = 416
    aug_type: str = 'default'  # 'default' or sannapersson'
    letterbox: bool = True
    fill: Tuple = (123.0, 117.0, 104.0)
    color_p: float = 0.4
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.7
    hue: float = 0.015
    blur_p: float = 0.1
    blur_size_min: int = 3
    blur_size_max: int = 7
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 2.0
    autocontrast_p: float = 0.1
    posterize_p: float = 0.1
    posterize_bits: int = 4
    grayscale_p: float = 0.1
    channelshuffle_p: float = 0.05
    perspective_p: float = 0.4
    perspective: float = 0.0
    translate: float = 0.1
    scale: float = 0.75
    shear_p: float = 0.4
    shear: float = 0.0  # unit: deg
    rotate_p: float = 0.4
    degrees: float = 0.0  # unit: deg
    crop_scale: float = 0.8
    ratio_min: float = 0.5
    ratio_max: float = 2.0
    flip_p: float = 0.5
    min_size: float= 1.0  # filter out too small boxes in augmented training data
    imgs_mean: Tuple = (0.485, 0.456, 0.406)
    imgs_std: Tuple = (0.229, 0.224, 0.225)
    multiscale_h: Tuple = (320, 352, 384, 416, 448, 480, 512, 544, 576, 608)  # should be multiple of max stride, (img_h,) to disable
    multiscale_w: Tuple = (320, 352, 384, 416, 448, 480, 512, 544, 576, 608)  # should be multiple of max stride, (img_w,) to disable


def voc_collate_fn(batch):
    xs, ys, borders = zip(*batch)
    for idx_img, y in enumerate(ys):
        y[:, 0] = idx_img
    return torch.stack(xs, dim=0), torch.cat(ys, dim=0), torch.stack(borders, dim=0)


class VocTrainDataLoader(DataLoader):
    def __init__(self, config: VocConfig, data_dir, batch_size, num_workers, collate_fn, shuffle=True, pin_memory=True,
                 nano=False):  # if True: use only the first five images as the entire dataset
        self.config = config
        if config.aug_type == 'default':
            transforms = v2.Compose([
                add_img_border_box,
                Resize(size=(config.img_h, config.img_w), letterbox=config.letterbox, fill=config.fill,
                       interpolation=InterpolationMode.BILINEAR, antialias=True),
                v2.RandomPerspective(distortion_scale=config.perspective, fill=config.fill,
                                     interpolation=InterpolationMode.BILINEAR),
                v2.RandomAffine(degrees=config.degrees, translate=(config.translate, config.translate),
                                scale=(1 - config.scale, 1 + config.scale * config.crop_scale),
                                shear=(-config.shear, config.shear, -config.shear, config.shear),
                                fill=config.fill, interpolation=InterpolationMode.BILINEAR),
                v2.ToImage(),
                v2.RandomResizedCrop(size=(config.img_h, config.img_w), scale=(config.crop_scale, 1.0),
                                     ratio=(config.ratio_min, config.ratio_max),
                                     interpolation=InterpolationMode.BILINEAR, antialias=True),
                v2.RandomApply([
                    v2.GaussianBlur(kernel_size=(config.blur_size_min, config.blur_size_max),
                                    sigma=(config.blur_sigma_min, config.blur_sigma_max)),
                ], p=config.blur_p),
                v2.RandomGrayscale(p=config.grayscale_p),
                v2.RandomAutocontrast(p=config.autocontrast_p),
                v2.ColorJitter(brightness=config.brightness, contrast=config.contrast,
                               saturation=config.saturation, hue=config.hue),
                v2.RandomHorizontalFlip(p=config.flip_p),
                v2.ClampBoundingBoxes(),
                v2.SanitizeBoundingBoxes(min_size=config.min_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
                Voc2Yolov3(),
            ])
        elif config.aug_type == 'sannapersson':
            transforms = v2.Compose([
                add_img_border_box,
                v2.RandomApply([
                    v2.ColorJitter(brightness=config.brightness, contrast=config.contrast,
                                   saturation=config.saturation, hue=config.hue),
                ], p=config.color_p),
                v2.RandomApply([
                    v2.GaussianBlur(kernel_size=(config.blur_size_min, config.blur_size_max),
                                    sigma=(config.blur_sigma_min, config.blur_sigma_max)),
                ], p=config.blur_p),
                v2.RandomAutocontrast(p=config.autocontrast_p),
                v2.RandomPosterize(bits=config.posterize_bits, p=config.posterize_p),
                v2.RandomGrayscale(p=config.grayscale_p),
                v2.RandomApply([v2.RandomChannelPermutation(),], p=config.channelshuffle_p),
                Resize(size=(config.img_h, config.img_w), letterbox=config.letterbox, fill=config.fill,
                       interpolation=InterpolationMode.BILINEAR, antialias=True),
                v2.RandomApply([
                    v2.RandomPerspective(distortion_scale=config.perspective, fill=config.fill,
                                         interpolation=InterpolationMode.BILINEAR),
                ], p=config.perspective_p),
                v2.RandomChoice([
                    v2.RandomApply([
                        v2.RandomAffine(degrees=0.0, translate=(config.translate, config.translate),
                                        scale=(1 - config.scale, 1 + config.scale * config.crop_scale),
                                        shear=(-config.shear, config.shear, -config.shear, config.shear),
                                        fill=config.fill, interpolation=InterpolationMode.BILINEAR),
                    ], p=config.shear_p),
                    v2.RandomApply([
                        v2.RandomAffine(degrees=config.degrees, translate=(config.translate, config.translate),
                                        scale=(1 - config.scale, 1 + config.scale * config.crop_scale),
                                        shear=None, fill=config.fill, interpolation=InterpolationMode.BILINEAR),
                    ], p=config.rotate_p),
                ]),
                v2.ToImage(),
                v2.RandomResizedCrop(size=(config.img_h, config.img_w), scale=(config.crop_scale, 1.0),
                                     ratio=(config.ratio_min, config.ratio_max),
                                     interpolation=InterpolationMode.BILINEAR, antialias=True),
                v2.RandomHorizontalFlip(p=config.flip_p),
                v2.ClampBoundingBoxes(),
                v2.SanitizeBoundingBoxes(min_size=config.min_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
                Voc2Yolov3(),
            ])
        else:
            raise ValueError(f"Unknown aug_type: {config.aug_type}")
        dataset_2007_trainval = torchvision.datasets.VOCDetection(root=data_dir, year='2007', image_set='trainval',
                                                                  download=False, transforms=transforms)
        dataset_2007_trainval_v2 = wrap_dataset_for_transforms_v2(dataset_2007_trainval, target_keys=['boxes', 'labels'])
        dataset_2012_trainval = torchvision.datasets.VOCDetection(root=data_dir, year='2012', image_set='trainval',
                                                                  download=False, transforms=transforms)
        dataset_2012_trainval_v2 = wrap_dataset_for_transforms_v2(dataset_2012_trainval, target_keys=['boxes', 'labels'])
        dataset = ConcatDataset([dataset_2007_trainval_v2, dataset_2012_trainval_v2])
        if nano:
            dataset = Subset(dataset, indices=range(5))
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                         num_workers=num_workers, pin_memory=pin_memory)


class VocValDataLoader(DataLoader):
    # Default shuffle=True since only eval partial data
    def __init__(self, config: VocConfig, data_dir, batch_size, num_workers, collate_fn, shuffle=True, pin_memory=True,
                 nano=False):  # if True: use only the first five images as the entire dataset
        self.config = config
        transforms = v2.Compose([
            add_img_border_box,
            Resize(size=(config.img_h, config.img_w), letterbox=config.letterbox, fill=config.fill, antialias=True),
            v2.ToImage(),
            v2.ClampBoundingBoxes(),
            v2.SanitizeBoundingBoxes(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
            Voc2Yolov3(),
        ])
        dataset_2007_test = torchvision.datasets.VOCDetection(root=data_dir, year='2007', image_set='test',
                                                              download=False, transforms=transforms)
        dataset = wrap_dataset_for_transforms_v2(dataset_2007_test, target_keys=['boxes', 'labels'])
        if nano:
            dataset = Subset(dataset, indices=range(5))
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                         num_workers=num_workers, pin_memory=pin_memory)


@dataclass
class BlankVocConfig:
    img_h: int = 416
    img_w: int = 416
    letterbox: bool = True
    fill: Tuple = (123.0, 117.0, 104.0)
    min_size: float = 1.0
    imgs_mean: Tuple = (0.485, 0.456, 0.406)
    imgs_std: Tuple = (0.229, 0.224, 0.225)


class BlankVocTrainDataLoader(DataLoader):
    """All images are set to zeros. Used for setting input-independent baseline."""
    def __init__(self, config: BlankVocConfig, data_dir, batch_size, num_workers, collate_fn, shuffle=True, pin_memory=True,
                 nano=False):  # if True: use only the first five images as the entire dataset
        self.config = config
        self.fill = {tv_tensors.Image: config.fill, "others": 0}
        transforms = v2.Compose([
            add_img_border_box,
            v2.ToImage(),
            v2.Lambda(
                lambda inp: tv_tensors.wrap(
                    torch.tensor(config.fill, dtype=inp.dtype, device=inp.device).view(3, 1, 1).expand(inp.shape),
                    like=inp)
                if isinstance(inp, tv_tensors.Image) else inp),
            Resize(size=(config.img_h, config.img_w), letterbox=config.letterbox, fill=self.fill, antialias=True),
            v2.ClampBoundingBoxes(),
            v2.SanitizeBoundingBoxes(min_size=config.min_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
            Voc2Yolov3(),
        ])
        dataset_2007_trainval = torchvision.datasets.VOCDetection(root=data_dir, year='2007', image_set='trainval',
                                                                  download=False, transforms=transforms)
        dataset_2007_trainval_v2 = wrap_dataset_for_transforms_v2(dataset_2007_trainval, target_keys=['boxes', 'labels'])
        dataset_2012_trainval = torchvision.datasets.VOCDetection(root=data_dir, year='2012', image_set='trainval',
                                                                  download=False, transforms=transforms)
        dataset_2012_trainval_v2 = wrap_dataset_for_transforms_v2(dataset_2012_trainval, target_keys=['boxes', 'labels'])
        dataset = ConcatDataset([dataset_2007_trainval_v2, dataset_2012_trainval_v2])
        if nano:
            dataset = Subset(dataset, indices=range(5))
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                         num_workers=num_workers, pin_memory=pin_memory)


class BlankVocValDataLoader(DataLoader):
    """All images are set to zeros. Used for setting input-independent baseline."""
    # Default shuffle=True since only eval partial data
    def __init__(self, config: BlankVocConfig, data_dir, batch_size, num_workers, collate_fn, shuffle=True, pin_memory=True,
                 nano=False):  # if True: use only the first five images as the entire dataset
        self.config = config
        self.fill = {tv_tensors.Image: config.fill, "others": 0}
        transforms = v2.Compose([
            add_img_border_box,
            v2.ToImage(),
            v2.Lambda(
                lambda inp: tv_tensors.wrap(
                    torch.tensor(config.fill, dtype=inp.dtype, device=inp.device).view(3, 1, 1).expand(inp.shape),
                    like=inp)
                if isinstance(inp, tv_tensors.Image) else inp),
            Resize(size=(config.img_h, config.img_w), letterbox=config.letterbox, fill=self.fill, antialias=True),
            v2.ClampBoundingBoxes(),
            v2.SanitizeBoundingBoxes(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
            Voc2Yolov3(),
        ])
        dataset_2007_test = torchvision.datasets.VOCDetection(root=data_dir, year='2007', image_set='test',
                                                              download=False, transforms=transforms)
        dataset = wrap_dataset_for_transforms_v2(dataset_2007_test, target_keys=['boxes', 'labels'])
        if nano:
            dataset = Subset(dataset, indices=range(5))
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                         num_workers=num_workers, pin_memory=pin_memory)



if __name__ == '__main__':
    # Test the dataloaders by `python -m dataloaders.voc` from the workspace directory
    import matplotlib.pyplot as plt
    data_dir = 'data/voc'
    config = VocConfig()
    dataloader_train = VocTrainDataLoader(config, data_dir, batch_size=32, num_workers=4, collate_fn=voc_collate_fn)
    dataloader_val = VocValDataLoader(config, data_dir, batch_size=32, shuffle=False, num_workers=4, collate_fn=voc_collate_fn)
    print(f"{len(dataloader_train)=}, {len(dataloader_train.dataset)=:,}")
    print(f"{len(dataloader_val)=}", f"{len(dataloader_val.dataset)=:,}")
    x, y = next(iter(dataloader_train))
    print(f"{x.shape=}; {y.shape=}")
    # Unnormalize the image for plotting
    img = x[0]
    img = img * torch.tensor(config.imgs_std).reshape(3, 1, 1) + torch.tensor(config.imgs_mean).reshape(3, 1, 1)
    print(f"{img.shape=}")
    print(f"{img.mean(dim=(1,2))=}, {img.std(dim=(1,2))=}")
    print(f"{img.min()=}, {img.max()=}")
    # plt.imshow(img.permute(1, 2, 0))
    # plt.show()
