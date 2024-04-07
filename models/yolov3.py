"""
Full definition of a YOLOv3 model, all of it in this single file.
Ref:
1) the official Darknet implementation:
https://github.com/AlexeyAB/darknet/blob/master/src/darknet.c
https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov3.cfg
https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov3-voc.cfg
2) the official YOLOv3 paper:
https://arxiv.org/abs/1804.02767
3) unofficial pytorch implementation:
https://github.com/ultralytics/yolov3
https://github.com/eriklindernoren/PyTorch-YOLOv3
"""

from pprint import pprint
import math
import inspect
from dataclasses import dataclass
from typing import Any, Callable, cast, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
import random
from copy import deepcopy

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops import box_iou, box_convert, clip_boxes_to_image, nms, batched_nms
import thop  # for FLOPs computation
from models.darknet53 import Darknet53Config, Darknet53Conv2d, Darknet53Backbone


class Concat(nn.Module):
    """
    Concatenate a list of tensors along dimension.
    Make torch.cat as a nn.Module to record it when visualizing the model structure.
    """
    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.d = dimension

    def forward(self, x: List[Tensor]) -> Tensor:
        return torch.cat(x, self.d)


@dataclass
class Yolov3Config:
    img_h: int = 416
    img_w: int = 416
    n_class: int = 80  # 80 for coco, 20 for voc
    n_scale: int = 3
    n_anchor_per_scale: int = 3
    anchors: Tuple[Tuple[Tuple[int, int], ...], ...] = (  # size(n_scale, n_anchor_per_scale, 2)
        ((10, 13), (16, 30), (33, 23)),  # scale3  (from stage3 & upsampled stage4 & upsampled stage5)
        ((30, 61), (62, 45), (59, 119)),  # scale4 (from stage4 & upsampled stage5)
        ((116, 90), (156, 198), (373, 326)),  # scale5 (from stage5)
    )  # w,h in pixels of a 416x416 image. IMPORTANT: from scale3 to scale5, order-aware!


class Yolov3Head(nn.Module):
    """
    Prediction head of YOLOv3.
    """
    def __init__(self, config: Yolov3Config) -> None:
        super().__init__()
        self.config = config

        # Along scale5 pass
        self.scale5_stage5_conv0 = Darknet53Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.scale5_stage5_conv1 = Darknet53Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.scale5_stage5_conv2 = Darknet53Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.scale5_stage5_conv3 = Darknet53Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.scale5_stage5_conv4 = Darknet53Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)  # pass output to scale4_stage5_conv4
        self.scale5_stage5_conv5 = Darknet53Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.scale5_stage5_conv6 = nn.Conv2d(1024, config.n_anchor_per_scale * (5 + config.n_class),
                                             kernel_size=1, stride=1, padding=0)
        # Along scale4 pass
        self.scale4_stage5_conv4 = Darknet53Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.scale4_stage5_upsample = nn.Upsample(scale_factor=2, mode='nearest')  # concat output with feat_stage4
        self.scale4_concat = Concat(dimension=1)
        self.scale4_stage4_conv0 = Darknet53Conv2d(768, 256, kernel_size=1, stride=1, padding=0)
        self.scale4_stage4_conv1 = Darknet53Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.scale4_stage4_conv2 = Darknet53Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.scale4_stage4_conv3 = Darknet53Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.scale4_stage4_conv4 = Darknet53Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # pass output to scale3_stage4_conv4
        self.scale4_stage4_conv5 = Darknet53Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.scale4_stage4_conv6 = nn.Conv2d(512, config.n_anchor_per_scale * (5 + config.n_class),
                                             kernel_size=1, stride=1, padding=0)
        # Along scale3 pass
        self.scale3_stage4_conv4 = Darknet53Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.scale3_stage4_upsample = nn.Upsample(scale_factor=2, mode='nearest')  # concat output with feat_stage3
        self.scale3_concat = Concat(dimension=1)
        self.scale3_stage3_conv0 = Darknet53Conv2d(384, 128, kernel_size=1, stride=1, padding=0)
        self.scale3_stage3_conv1 = Darknet53Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.scale3_stage3_conv2 = Darknet53Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.scale3_stage3_conv3 = Darknet53Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.scale3_stage3_conv4 = Darknet53Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.scale3_stage3_conv5 = Darknet53Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.scale3_stage3_conv6 = nn.Conv2d(256, config.n_anchor_per_scale * (5 + config.n_class),
                                             kernel_size=1, stride=1, padding=0)


    def forward(self, feat_stage3: Tensor, feat_stage4: Tensor, feat_stage5: Tensor) -> Tuple[Tensor, ...]:  # TODO: correct type hints
        """
        Args:
            feat_stage3 (Tensor): (N, 256, img_h / 8, img_w / 8)
            feat_stage4 (Tensor): (N, 512, img_h / 16, img_w / 16)
            feat_stage5 (Tensor): (N, 1024, img_h / 32, img_w / 32)
        Returns:
            logit_scale3 (Tensor): (N, n_anchor_per_scale, img_h / 8, img_w / 8, 5 + n_class)
            logit_scale4 (Tensor): (N, n_anchor_per_scale, img_h / 16, img_w / 16, 5 + n_class)
            logit_scale5 (Tensor): (N, n_anchor_per_scale, img_h / 32, img_w / 32, 5 + n_class)
        """
        # Along scale5 pass
        # N x 1024 x img_h / 32 x img_w / 32
        x = self.scale5_stage5_conv0(feat_stage5)
        # N x 512 x img_h / 32 x img_w / 32
        x = self.scale5_stage5_conv1(x)
        # N x 1024 x img_h / 32 x img_w / 32
        x = self.scale5_stage5_conv2(x)
        # N x 512 x img_h / 32 x img_w / 32
        x = self.scale5_stage5_conv3(x)
        # N x 1024 x img_h / 32 x img_w / 32
        feat_scale5_stage5_conv4 = self.scale5_stage5_conv4(x)  # also pass feat_scale5_stage5_conv4 to scale4_stage5_conv4
        # N x 512 x img_h / 32 x img_w / 32
        x = self.scale5_stage5_conv5(feat_scale5_stage5_conv4)
        # N x 1024 x img_h / 32 x img_w / 32
        logit_scale5 = self.scale5_stage5_conv6(x)
        # N x n_anchor_per_scale * (5 + n_class) x img_h / 32 x img_w / 32
        batch_size, _, n_cell_h_scale5, n_cell_w_scale5 = logit_scale5.shape
        logit_scale5 = logit_scale5.view(
            batch_size, self.config.n_anchor_per_scale, 5 + self.config.n_class, n_cell_h_scale5, n_cell_w_scale5
        ).permute(0, 1, 3, 4, 2).contiguous()
        # N x n_anchor_per_scale x img_h / 32 x img_w / 32 x (5 + n_class)

        # Along scale4 pass
        # N x 512 x img_h / 32 x img_w / 32
        x = self.scale4_stage5_conv4(feat_scale5_stage5_conv4)
        # N x 256 x img_h / 32 x img_w / 32
        x = self.scale4_stage5_upsample(x)
        # x: N x 256 x img_h / 16 x img_w / 16; feat_stage4: N x 512 x img_h / 16 x img_w / 16
        x = self.scale4_concat([x, feat_stage4])
        # N x 768 x img_h / 16 x img_w / 16
        x = self.scale4_stage4_conv0(x)
        # N x 256 x img_h / 16 x img_w / 16
        x = self.scale4_stage4_conv1(x)
        # N x 512 x img_h / 16 x img_w / 16
        x = self.scale4_stage4_conv2(x)
        # N x 256 x img_h / 16 x img_w / 16
        x = self.scale4_stage4_conv3(x)
        # N x 512 x img_h / 16 x img_w / 16
        feat_scale4_stage4_conv4 = self.scale4_stage4_conv4(x) # also pass feat_scale4_stage4_conv4 to scale3_stage4_conv4
        # N x 256 x img_h / 16 x img_w / 16
        x = self.scale4_stage4_conv5(feat_scale4_stage4_conv4)
        # N x 512 x img_h / 16 x img_w / 16
        logit_scale4 = self.scale4_stage4_conv6(x)
        # N x n_anchor_per_scale * (5 + n_class) x img_h / 16 x img_w / 16
        batch_size, _, n_cell_h_scale4, n_cell_w_scale4 = logit_scale4.shape
        logit_scale4 = logit_scale4.view(
            batch_size, self.config.n_anchor_per_scale, 5 + self.config.n_class, n_cell_h_scale4, n_cell_w_scale4
        ).permute(0, 1, 3, 4, 2).contiguous()
        # N x n_anchor_per_scale x img_h / 16 x img_w / 16 x (5 + n_class)

        # Along scale3 pass
        # N x 256 x img_h / 16 x img_w / 16
        x = self.scale3_stage4_conv4(feat_scale4_stage4_conv4)
        # N x 128 x img_h / 16 x img_w / 16
        x = self.scale3_stage4_upsample(x)
        # x: N x 128 x img_h / 8 x img_w / 8; feat_stage3: N x 256 x img_h / 8 x img_w / 8
        x = self.scale3_concat([x, feat_stage3])
        # N x 384 x img_h / 8 x img_w / 8
        x = self.scale3_stage3_conv0(x)
        # N x 128 x img_h / 8 x img_w / 8
        x = self.scale3_stage3_conv1(x)
        # N x 256 x img_h / 8 x img_w / 8
        x = self.scale3_stage3_conv2(x)
        # N x 128 x img_h / 8 x img_w / 8
        x = self.scale3_stage3_conv3(x)
        # N x 256 x img_h / 8 x img_w / 8
        x = self.scale3_stage3_conv4(x)
        # N x 128 x img_h / 8 x img_w / 8
        x = self.scale3_stage3_conv5(x)
        # N x 256 x img_h / 8 x img_w / 8
        logit_scale3 = self.scale3_stage3_conv6(x)
        # N x n_anchor_per_scale * (5 + n_class) x img_h / 8 x img_w / 8
        batch_size, _, n_cell_h_scale3, n_cell_w_scale3 = logit_scale3.shape
        logit_scale3 = logit_scale3.view(
            batch_size, self.config.n_anchor_per_scale, 5 + self.config.n_class, n_cell_h_scale3, n_cell_w_scale3
        ).permute(0, 1, 3, 4, 2).contiguous()
        # N x n_anchor_per_scale x img_h / 8 x img_w / 8 x (5 + n_class)

        return logit_scale3, logit_scale4, logit_scale5


class Yolov3(nn.Module):
    """
    Yolov3 detection model.
    """
    def __init__(self, config: Yolov3Config) -> None:
        super().__init__()
        self.config = config
        self.backbone = Darknet53Backbone(Darknet53Config())
        self.head = Yolov3Head(config)

        # Build strides, anchors
        forward = lambda imgs: self.forward(imgs)
        self.stride_scale3, self.stride_scale4, self.stride_scale5 = [
            config.img_w / logit.shape[-2] for logit in forward(torch.zeros(1, 3, config.img_h, config.img_w))
        ]  # 8., 16., 32. by default
        # Register anchors as buffer to make they switch device with model
        self.register_buffer("anchors_scale3",  # size(n_anchor_per_scale, 2), w,h in unit of cell
                             torch.tensor(config.anchors[0], dtype=torch.float32) / self.stride_scale3)
        self.register_buffer("anchors_scale4",  # size(n_anchor_per_scale, 2), w,h in unit of cell
                             torch.tensor(config.anchors[1], dtype=torch.float32) / self.stride_scale4)
        self.register_buffer("anchors_scale5",  # size(n_anchor_per_scale, 2), w,h in unit of cell
                             torch.tensor(config.anchors[2], dtype=torch.float32) / self.stride_scale5)
        # Init grid, anchor_grid
        self.grid_scale3, self.anchor_grid_scale3 = torch.empty(0), torch.empty(0)
        self.grid_scale4, self.anchor_grid_scale4 = torch.empty(0), torch.empty(0)
        self.grid_scale5, self.anchor_grid_scale5 = torch.empty(0), torch.empty(0)

        # Init all weights & biases
        self.apply(self._init_weights)
        # Apply special init to last conv layers for detection logits
        self._init_biases()

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            pass  # torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(module, nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03
        elif isinstance(module, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
            module.inplace = True


    def _init_biases(self):
        """
        Initialize biases into last conv layers for detection logits.
        """
        for detect_conv, stride in ((self.head.scale3_stage3_conv6, self.stride_scale3),
                                    (self.head.scale4_stage4_conv6, self.stride_scale4),
                                    (self.head.scale5_stage5_conv6, self.stride_scale5)):
            b = detect_conv.bias.view(self.config.n_anchor_per_scale, -1)  # n_anchor_per_scale x (5 + n_class)
            b.data[:, 4] += math.log(8 / (640 / stride) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + self.config.n_class] += (math.log(0.6 / (self.config.n_class - 0.99999)))  # cls
            detect_conv.bias = nn.Parameter(b.view(-1), requires_grad=True)


    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


    def estimate_mfu(self):
        """
        Estimate model flops utilization (MFU).
        """
        n_param = sum(p.numel() for p in self.parameters())  # number parameters
        n_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)  # number gradients
        print(f"{'layer':>5} {'name':>42} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(self.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %42s %9s %12g %20s %10.3g %10.3g"
                % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std())
            )

        # FLOPs calculation
        p = next(self.parameters())
        stride_max = int(self.stride_scale5)  # max stride
        img = torch.empty((1, p.shape[1], stride_max, stride_max), device=p.device)  # input image in BCHW format
        flops = thop.profile(deepcopy(self), inputs=(img,), verbose=False)[0] / 1e9 * 2  # max stride GFLOPs
        fs = f", {flops * self.config.img_h / stride_max * self.config.img_w / stride_max:.1f} GFLOPs"  # img_h x img_w GFLOPs

        print(f"YOLOv3 summary: {len(list(self.modules()))} layers, {n_param} parameters, {n_grad} gradients{fs}")


    def forward(self, imgs: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, ...]:
        """
        Forward pass of single-scale training.
        Args:
            imgs (Tensor): size(N, 3, img_h, img_w)
            targets (Tensor): TODO
        Returns:
            logit_scale3 (Tensor): size(N, n_anchor_per_scale, img_h / 8, img_w / 8, 5 + n_class)
                logit[i, j, k, l, 0:4] is the box coordinates for the j-th box in the k,l-th cell,
                                        i.e., t_x, t_y, t_w, t_h in the paper
                logit[i, j, k, l, 4] is the objectness confidence score for the j-th box in the k,l-th cell,
                                      i.e., t_o in the paper
                logit[i, j, k, l, 5:5+n_class] is the class logit (before softmax) for the j-th box in the k,l-th cell
            logit_scale4 (Tensor): size(N, n_anchor_per_scale, img_h / 16, img_w / 16, 5 + n_class)
            logit_scale5 (Tensor): size(N, n_anchor_per_scale, img_h / 32, img_w / 32, 5 + n_class)
            loss (Tensor): size(,), weighted sum of the following losses
            TODO: components of the loss
        """
        device = imgs.device

        # FUTURE: visualize feature maps as https://github.com/ultralytics/yolov3/blob/master/utils/plots.py#feature_visualization
        # FUTURE: fuse conv and bn as https://github.com/ultralytics/yolov3/blob/master/models/yolo.py#BaseModel
        # Forward the Yolov3 model itself
        # N x 3 x img_h x img_w
        feat_stage3, feat_stage4, feat_stage5 = self.backbone(imgs)
        # feat_stage3: N x 256 x img_h / 8 x img_w / 8
        # feat_stage4: N x 512 x img_h / 16 x img_w / 16
        # feat_stage5: N x 1024 x img_h / 32 x img_w / 32
        logit_scale3, logit_scale4, logit_scale5 = self.head(feat_stage3, feat_stage4, feat_stage5)
        # logit_scale3: N x n_anchor_per_scale x img_h / 8 x img_w / 8 x (5 + n_class)
        # logit_scale4: N x n_anchor_per_scale x img_h / 16 x img_w / 16 x (5 + n_class)
        # logit_scale5: N x n_anchor_per_scale x img_h / 32 x img_w / 32 x (5 + n_class)

        # TODO: compute loss
        return logit_scale3, logit_scale4, logit_scale5


    @torch.inference_mode()
    def generate(self, imgs: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, ...]:
        """
        Forward pass of single-scale inference.
        Args:
            imgs (Tensor): size(N, 3, img_h, img_w)
            targets (Tensor): TODO
        Returns:
            result (Tensor): (N, n_anchor_per_scale * img_h / 8 * img_w / 8
                                 + n_anchor_per_scale * img_h / 16 * img_w / 16
                                 + n_anchor_per_scale * img_h / 32 * img_w / 32, 5 + n_class)
                             restored x,y,w,h,conf,prob_class from logits (coordinates are in pixels)
            logit_scale3 (Tensor): size(N, n_anchor_per_scale, img_h / 8, img_w / 8, 5 + n_class)
                logit[i, j, k, l, 0:4] is the box coordinates for the j-th box in the k,l-th cell,
                                        i.e., t_x, t_y, t_w, t_h in the paper
                logit[i, j, k, l, 4] is the objectness confidence score for the j-th box in the k,l-th cell,
                                      i.e., t_o in the paper
                logit[i, j, k, l, 5:5+n_class] is the class logit (before softmax) for the j-th box in the k,l-th cell
            logit_scale4 (Tensor): size(N, n_anchor_per_scale, img_h / 16, img_w / 16, 5 + n_class)
            logit_scale5 (Tensor): size(N, n_anchor_per_scale, img_h / 32, img_w / 32, 5 + n_class)
            loss (Tensor): size(,), weighted sum of the following losses
            TODO: components of the loss
        """
        logit_scale3, logit_scale4, logit_scale5 = self.forward(imgs, targets)  # TODO: get loss
        result = []  # inference output

        # For scale3
        batch_size, _, n_cell_h_scale3, n_cell_w_scale3, _ = logit_scale3.shape
        if self.grid_scale3.shape[2:4] != (n_cell_h_scale3, n_cell_w_scale3):
            self.grid_scale3, self.anchor_grid_scale3 = self._make_grid(
                n_cell_h_scale3, n_cell_w_scale3, self.anchors_scale3, self.stride_scale3
            )
        xy_scale3, wh_scale3, conf_scale3, prob_class_scale3 = torch.split(
            torch.sigmoid(logit_scale3), [2, 2, 1, self.config.n_class], dim=-1
        )
        xy_scale3 = (xy_scale3 * 2 + self.grid_scale3) * self.stride_scale3  # xy
        wh_scale3 = (wh_scale3 * 2) ** 2 * self.anchor_grid_scale3  # wh
        result_scale3 = torch.cat([xy_scale3, wh_scale3, conf_scale3, prob_class_scale3], dim=-1)
        result.append(result_scale3.view(
            batch_size, self.config.n_anchor_per_scale * n_cell_h_scale3 * n_cell_w_scale3, 5 + self.config.n_class
        ))

        # For scale4
        batch_size, _, n_cell_h_scale4, n_cell_w_scale4, _ = logit_scale4.shape
        if self.grid_scale4.shape[2:4] != (n_cell_h_scale4, n_cell_w_scale4):
            self.grid_scale4, self.anchor_grid_scale4 = self._make_grid(
                n_cell_h_scale4, n_cell_w_scale4, self.anchors_scale4, self.stride_scale4
            )
        xy_scale4, wh_scale4, conf_scale4, prob_class_scale4 = torch.split(
            torch.sigmoid(logit_scale4), [2, 2, 1, self.config.n_class], dim=-1
        )
        xy_scale4 = (xy_scale4 * 2 + self.grid_scale4) * self.stride_scale4  # xy
        wh_scale4 = (wh_scale4 * 2) ** 2 * self.anchor_grid_scale4  # wh
        result_scale4 = torch.cat([xy_scale4, wh_scale4, conf_scale4, prob_class_scale4], dim=-1)
        result.append(result_scale4.view(
            batch_size, self.config.n_anchor_per_scale * n_cell_h_scale4 * n_cell_w_scale4, 5 + self.config.n_class
        ))

        # For scale5
        batch_size, _, n_cell_h_scale5, n_cell_w_scale5, _ = logit_scale5.shape
        if self.grid_scale5.shape[2:4] != (n_cell_h_scale5, n_cell_w_scale5):
            self.grid_scale5, self.anchor_grid_scale5 = self._make_grid(
                n_cell_h_scale5, n_cell_w_scale5, self.anchors_scale5, self.stride_scale5
            )
        xy_scale5, wh_scale5, conf_scale5, prob_class_scale5 = torch.split(
            torch.sigmoid(logit_scale5), [2, 2, 1, self.config.n_class], dim=-1
        )
        xy_scale5 = (xy_scale5 * 2 + self.grid_scale5) * self.stride_scale5  # xy = (2.0 * sigmoid(t_xy) - 0.5 + c_xy) * stride
        wh_scale5 = (wh_scale5 * 2) ** 2 * self.anchor_grid_scale5  # wh = (2.0 * sigmoid(t_wh)) ** 2 * p_wh
        result_scale5 = torch.cat([xy_scale5, wh_scale5, conf_scale5, prob_class_scale5], dim=-1)
        result.append(result_scale5.view(
            batch_size, self.config.n_anchor_per_scale * n_cell_h_scale5 * n_cell_w_scale5, 5 + self.config.n_class
        ))

        return torch.cat(result, dim=1), logit_scale3, logit_scale4, logit_scale5  # TODO: return loss


    def _make_grid(self, n_cell_h: int, n_cell_w: int, anchors_per_scale: Tensor, stride: float) -> Tuple[Tensor, ...]:
        """
        Make grid and anchor_grid for a scale.
        Args:
            n_cell_h (int): number of cells along height
            n_cell_w (int): number of cells along width
            anchors_per_scale (Tensor): size(n_anchor_per_scale, 2), w,h in unit of cell
            stride (float): stride of the scale
        Returns:
            grid (Tensor): size(1, n_anchor_per_scale, n_cell_h, n_cell_w, 2), grid offset in unit of cell
            anchor_grid (Tensor): size(1, n_anchor_per_scale, n_cell_h, n_cell_w, 2), anchor size in pixels
        """
        device, dtype = anchors_per_scale.device, anchors_per_scale.dtype
        shape = (1, self.config.n_anchor_per_scale, n_cell_h, n_cell_w, 2)  # grid shape
        cell_y, cell_x = torch.meshgrid(
            torch.arange(n_cell_h, dtype=dtype, device=device),
            torch.arange(n_cell_w, dtype=dtype, device=device),
            indexing='ij'
        )  # size(n_cell_h, n_cell_w)
        grid = torch.stack((cell_x, cell_y), dim=-1).expand(shape) - 0.5  # add grid offset, i.e. xy_scale = 2.0 * sigmoid(t_xy) - 0.5 + c_xy
        anchor_grid = (anchors_per_scale * stride).view((1, self.config.n_anchor_per_scale, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


if __name__ == '__main__':
    # Test the model by `python -m models.yolov3` from the workspace directory
    # TODO: change following back to for Yolov3 with loss computation
    config = Yolov3Config()
    model = Yolov3(config)
    # print(model)
    model.estimate_mfu()
    print(f"num params: {model.get_num_params():,}")  # 61,949,149

    imgs = torch.randn((2, 3, config.img_h, config.img_w))

    logit_scale3, logit_scale4, logit_scale5 = model(imgs)
    print(f"\n{logit_scale3.shape=}\n{logit_scale4.shape=}\n{logit_scale5.shape=}\n")

    result, logit_scale3, logit_scale4, logit_scale5 = model.generate(imgs)
    print(f"{result.shape=}\n{logit_scale3.shape=}\n{logit_scale4.shape=}\n{logit_scale5.shape=}\n")
