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
from torchvision.ops import box_iou, box_convert, clip_boxes_to_image, nms, batched_nms, complete_box_iou_loss
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
    match_thresh: float = 4.0  # iou or wh ratio threshold to match a target to an anchor when calculating loss
    rescore: float = 1.0  # 0.0~1.0, rescore ratio; if 0.0, use 1 as objectness target; if 1.0, use iou as objectness target
    smooth: float = 0.0  # 0.0~1.0, smooth ratio for class BCE loss; 0.0 for no smoothing; 0.1 is a common choice
    pos_weight_class: float = 1.0  # weight for class BCE loss of positive examples, as if the positive examples are duplicated
    pos_weight_obj: float = 1.0  # weight for obj BCE loss of positive examples, as if the positive examples are duplicated
    balance: Tuple[float, float, float] = (4.0, 1.0, 0.4)  # balance weights for scale3~5 loss_obj. Maybe it's because the scale3 has less portion of objects and larger portion of no-obj cells, so if the no-obj loss is near zero, then obj loss of scale3 is less due to mean reduction, which is why balance_scale3 is larger, to ensures that the predictions at different scales contribute appropriately to the total loss.
    lambda_box: float = 0.05  # weight for box loss
    lambda_obj: float = 1.0  # weight for obj loss
    lambda_class: float = 0.5  # weight for class loss
    score_thresh: float = 0.001  # threshold for (objectness score * class probability) when filtering inference results
    iou_thresh: float = 0.6  # NMS iou threshold when filtering inference results


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
            config.img_w / logit.shape[-2] for logit in forward(torch.zeros(1, 3, config.img_h, config.img_w))[:3]
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
        # Balance for obj loss of different scales
        self.balance_scale3, self.balance_scale4, self.balance_scale5 = config.balance

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


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError("FUTURE: init from pretrained model")


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


    def forward(self, imgs: Tensor, target: Optional[Tensor] = None) -> Tuple[Tensor, ...]:
        """
        Forward pass of single-scale training.
        Args:
            imgs (Tensor): size(N, 3, img_h, img_w)
            target (Tensor): size(n_batch_obj, 6), n_batch_obj is the total number of object boxes in the batch (N imgs)
                target[i, 0] is the idx of the image in the batch, 0~batch_size-1
                target[i, 1] is the class index for a object box, 0.0~float(n_class-1), no background class
                target[i, 2:6] is the box coordinates for a object box, normalized by img w,h, CXCYWH format
        Returns:
            logit_scale3 (Tensor): size(N, n_anchor_per_scale, img_h / 8, img_w / 8, 5 + n_class)
                logit[i, j, k, l, 0:4] is the box coordinates for the j-th box in the k,l-th cell,
                                       i.e., t_x, t_y, t_w, t_h in the paper
                logit[i, j, k, l, 4] is the objectness confidence score for the j-th box in the k,l-th cell,
                                     i.e., t_o in the paper
                logit[i, j, k, l, 5:5+n_class] is the class logit (before softmax) for the j-th box in the k,l-th cell
            logit_scale4 (Tensor): size(N, n_anchor_per_scale, img_h / 16, img_w / 16, 5 + n_class)
            logit_scale5 (Tensor): size(N, n_anchor_per_scale, img_h / 32, img_w / 32, 5 + n_class)
            loss (Tensor): size(,), combination of the following losses
            loss_obj (Tensor): size(,), BCE loss for objectness
            loss_class (Tensor): size(,), (smooth-)BCE or focal loss for class
            loss_box (Tensor): size(,), CIoU loss for box
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

        if target is not None:
            # If we are given some desired target also calculate the loss
            loss, loss_obj, loss_class, loss_box = self._compute_loss(logit_scale3, logit_scale4, logit_scale5, target)
        else:
            loss, loss_obj, loss_class, loss_box = None, None, None, None

        return logit_scale3, logit_scale4, logit_scale5, loss, loss_obj, loss_class, loss_box


    def _compute_loss(self, logit_scale3: Tensor, logit_scale4: Tensor, logit_scale5: Tensor, target: Tensor) -> Tuple[Tensor, ...]:
        """
        Compute the loss and its components.
        Args:
            logit_scale3 (Tensor): size(N, n_anchor_per_scale, img_h / 8, img_w / 8, 5 + n_class)
            logit_scale4 (Tensor): size(N, n_anchor_per_scale, img_h / 16, img_w / 16, 5 + n_class)
            logit_scale5 (Tensor): size(N, n_anchor_per_scale, img_h / 32, img_w / 32, 5 + n_class)
            target (Tensor): size(n_batch_obj, 6)
        Returns:
            loss (Tensor): size(,)
            loss_obj (Tensor): size(,)
            loss_class (Tensor): size(,)
            loss_box (Tensor): size(,)
        """
        batch_size, dtype, device = logit_scale3.shape[0], logit_scale3.dtype, logit_scale3.device
        loss = torch.tensor(0.0, dtype=dtype, device=device)
        loss_obj = torch.tensor(0.0, dtype=dtype, device=device)
        loss_class = torch.tensor(0.0, dtype=dtype, device=device)
        loss_box = torch.tensor(0.0, dtype=dtype, device=device)

        # Iterate over scales
        for logit, anchors, balance in ((logit_scale3, self.anchors_scale3, self.balance_scale3),
                                        (logit_scale4, self.anchors_scale4, self.balance_scale4),
                                        (logit_scale5, self.anchors_scale5, self.balance_scale5)):
            target_class, target_box, responsible_idx, responsible_anchor = self._build_target(logit, target, anchors)  # size(n_responsible,)
            idx_img, idx_anchor, cell_y, cell_x = responsible_idx.T  # size(n_responsible,)
            n_responsible = idx_img.shape[0]
            # Init objectness target to zeros (not responsible)
            target_obj = torch.zeros_like(logit[:, :, :, :, 4])  # size(N, n_anchor_per_scale, n_cell_h, n_cell_w)

            if n_responsible > 0:
                # Responsible coord & class logit
                responsible_logit = logit[idx_img, idx_anchor, cell_y, cell_x]  # size(n_responsible, 5 + n_class)
                logit_xy, logit_wh, _, logit_class = responsible_logit.split(
                    [2, 2, 1, self.config.n_class], dim=-1
                )  # size(n_responsible, 2), size(n_responsible, 2), size(n_responsible, 1), size(n_responsible, n_class)

                # Regression - box CIoU loss
                # Restore logits to predictions, in unit of cell w,h, relative to top-left of responsible cell
                pred_xy = torch.sigmoid(logit_xy) * 2 - 0.5  # size(n_responsible, 2)
                pred_wh = (torch.sigmoid(logit_wh) * 2) ** 2 * responsible_anchor  # size(n_responsible, 2)
                pred_box = torch.cat((pred_xy, pred_wh), dim=-1)  # size(n_responsible, 4)
                # Compute iou between predictions and targets
                iou_loss = complete_box_iou_loss(  # iou_loss = 1 - iou  # FUTURE: try other types of iou loss
                    box_convert(pred_box, in_fmt='cxcywh', out_fmt='xyxy'),  # size(n_responsible, 4)
                    box_convert(target_box, in_fmt='cxcywh', out_fmt='xyxy'),  # size(n_responsible, 4)
                    reduction='none'
                )  # size(n_responsible,)
                loss_box += iou_loss.mean()

                # Objectness - assign objectness target to responsible pred
                iou = 1.0 - iou_loss.detach().clamp(min=0).to(dtype)  # size(n_responsible,), objectness target for responsible pred
                if self.config.rescore < 1.0:
                    iou = (1.0 - self.config.rescore) + iou * self.config.rescore
                target_obj[idx_img, idx_anchor, cell_y, cell_x] = iou

                # Classification - class smooth-BCE-with-logits loss  # FUTURE: try CE loss for voc, since indep class; try focal loss
                target_class = F.one_hot(target_class, self.config.n_class).to(dtype)
                # Optional label smoothing for BCE: positive class: 1 - 0.5 * smooth, negative class: 0.5 * smooth
                target_class = target_class * (1.0 - self.config.smooth) + 0.5 * self.config.smooth
                loss_class += F.binary_cross_entropy_with_logits(
                    logit_class, target_class,
                    reduction='mean', pos_weight=torch.tensor(self.config.pos_weight_class, device=device)
                )

            # Objectness - obj BCE loss  # FUTURE: autobalance, but it's disabled in YOLOv5, so maybe no good
            loss_obj += balance * F.binary_cross_entropy_with_logits(
                logit[:, :, :, :, 4], target_obj,
                reduction='mean', pos_weight=torch.tensor(self.config.pos_weight_obj, device=device)
            )

        loss_box *= self.config.lambda_box
        loss_obj *= self.config.lambda_obj
        loss_class *= self.config.lambda_class
        loss = batch_size * (loss_box + loss_obj + loss_class)

        return loss, loss_obj.detach(), loss_class.detach(), loss_box.detach()


    def _build_target(self, logit: Tensor, target: Tensor, anchors: Tensor) -> Tuple[Tensor, ...]:
        """
        Build target for the loss computation of a scale.
        Args:
            logit (Tensor): size(N, n_anchor_per_scale, img_h / stride, img_w / stride, 5 + n_class)
            target (Tensor): size(n_batch_obj, 6)
            anchors (Tensor): size(n_anchor_per_scale, 2), w,h in unit of cell
        Returns:
            target_class (Tensor): size(n_responsible,)
            target_box (Tensor): size(n_responsible, 4), x,y,w,h in unit of cell, relative to top-left of responsible cell
            responsible_idx (Tensor): size(n_responsible, 4),torch.int64
                responsible_idx[i, 0] is the idx_img in this batch
                responsible_idx[i, 1] idx of responsible anchor of i'th responsible cell in this scale
                responsible_idx[i, 2] idx_y of i'th responsible cell
                responsible_idx[i, 3] idx_x of i'th responsible cell
            responsible_anchor (Tensor): size(n_responsible, 2), anchors of responsible logits, w,h in unit of cell
        """
        dtype, device = target.dtype, target.device
        n_batch_obj = target.shape[0]
        idx_anchor_per_scale = torch.arange(
            self.config.n_anchor_per_scale, dtype=dtype, device=device
        ).repeat_interleave(n_batch_obj)  # size(n_anchor_per_scale * n_batch_obj,)
        target = torch.cat((  # append anchor idx
            target.repeat(self.config.n_anchor_per_scale, 1, 1),  # size(n_anchor_per_scale, n_batch_obj, 6)
            idx_anchor_per_scale.view(self.config.n_anchor_per_scale, n_batch_obj, 1)  # size(n_anchor_per_scale, n_batch_obj, 1)
        ), dim=-1)  # size(n_anchor_per_scale, n_batch_obj, 7)
        n_cell_h, n_cell_w = logit.shape[2:4]
        # Normalize box coordinates into unit of cell
        target[:, :, 2:6] = target[:, :, 2:6] * torch.tensor([n_cell_w, n_cell_h, n_cell_w, n_cell_h], device=device)
        # Match target to anchors
        if n_batch_obj > 0:
            # Match  # FUTURE: how about match by iou
            wh_ratio = target[:, :, 4:6] / anchors.view(self.config.n_anchor_per_scale, 1, 2)  # size(n_anchor_per_scale, n_batch_obj, 2)
            match_score = torch.max(torch.max(wh_ratio, 1.0 / wh_ratio), dim=-1).values  # size(n_anchor_per_scale, n_batch_obj)
            mask_matched = match_score < self.config.match_thresh
            target = target[mask_matched]  # size(n_matched, 7)
            n_matched = target.shape[0]
            # If box is near a border of two cells (exclude img edge),
            # make anchors in nearby cells also responsible.  # FUTURE: won't this make NMS more burdensome?
            target_xy = target[:, 2:4]  # size(n_matched, 2)
            target_xy_inv = torch.tensor([n_cell_w, n_cell_h], device=device) - target_xy  # size(n_matched, 2)
            near_border_right, near_border_bottom = ((target_xy % 1 < 0.5) & (target_xy > 1)).T  # size(n_matched,), bool
            near_border_left, near_border_top = ((target_xy_inv % 1 < 0.5) & (target_xy_inv > 1)).T  # size(n_matched,), bool
            mask_responsible = torch.stack((
                torch.ones_like(near_border_right),  # original x,y
                near_border_right,
                near_border_bottom,
                near_border_left,
                near_border_top,
            ))  # size(5, n_matched), bool
            target = target.repeat(5, 1, 1)  # size(5, n_matched, 7)
            # Dilate the responsible cells for near-cell-border logits
            target = target[mask_responsible]  # size(n_responsible, 7)
            offset = torch.tensor([  # dilation direction
                [[0.0, 0.0]],  # original x,y, original cell is always responsible
                [[-0.5, 0.0]],  # near the right of a border, make the left cell also responsible
                [[0.0, -0.5]],  # near the bottom of a border, make the top cell also responsible
                [[0.5, 0.0]],  # near the left of a border, make the right cell also responsible
                [[0.0, 0.5]],  # near the top of a border, make the bottom cell also responsible
            ], dtype=dtype, device=device)  # size(5, 1, 2)
            offset = offset.expand(5, n_matched, 2)  # size(5, n_matched, 2)
            offset = offset[mask_responsible]  # size(n_responsible, 2)
        else:
            target = target[0]  # size(0, 7)
            offset = 0
        # Convert to formats for loss computation
        target_class = target[:, 1].to(torch.int64)  # size(n_responsible,)
        target_xy = target[:, 2:4]  # size(n_responsible, 2)
        target_wh = target[:, 4:6]  # size(n_responsible, 2)
        cell_xy = (target_xy + offset).to(torch.int64)  # size(n_responsible, 2), cell idx of responsible cells
        cell_x, cell_y = cell_xy.T  # size(n_responsible,)
        cell_x = cell_x.clamp(0, n_cell_w - 1)  # size(n_responsible,)
        cell_y = cell_y.clamp(0, n_cell_h - 1)  # size(n_responsible,)
        target_box = torch.cat((
            target_xy - cell_xy,  # size(n_responsible, 2), x,y relative to top-left of responsible cell, in unit of cell w,h
            target_wh,  # size(n_responsible, 2)
        ), dim=-1)  # size(n_responsible, 4)
        idx_img = target[:, 0].to(torch.int64)  # size(n_responsible,)
        idx_anchor = target[:, 6].to(torch.int64)  # size(n_responsible,)
        responsible_idx = torch.stack((idx_img, idx_anchor, cell_y, cell_x), dim=-1)  # size(n_responsible, 4)
        responsible_anchor = anchors[idx_anchor]  # size(n_responsible, 2), w,h in unit of cell

        return target_class, target_box, responsible_idx, responsible_anchor


    def configure_optimizers(self, optimizer_type, learning_rate, betas, weight_decay, device_type, use_fused):
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups. any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls decay, all biases and norms don't.
        # FUTURE: divide nodecay_params into two groups: bias & norm weight, so that bias lr can falls from warmup_bias_lr to lr during warmup
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        if optimizer_type == 'adamw':
            # Create AdamW optimizer and use the fused version if it is available
            if use_fused:
                fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
                use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")
        elif optimizer_type == 'adam':
            # Create Adam optimizer and use the fused version if it is available
            if use_fused:
                fused_available = 'fused' in inspect.signature(torch.optim.Adam).parameters
                use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.Adam(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused Adam: {use_fused}")
        elif optimizer_type == 'sgd':
            # Create SGD optimizer
            optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, momentum=betas[0], nesterov=True)
            print(f"using SGD")
        else:
            raise ValueError(f"unrecognized optimizer_type: {optimizer_type}")

        return optimizer


    @torch.inference_mode()
    def generate(self, imgs: Tensor, target: Optional[Tensor] = None, border: Optional[Tensor] = None) -> Tuple[Tensor, ...]:
        """
        Forward pass of single-scale inference.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Args:
            imgs (Tensor): size(N, 3, img_h, img_w)
            target (Tensor): size(n_batch_obj, 6)
            border (Tensor): size(N, 4), torch.int64, border of non-pad images, x1,y1,x2,y2 in pixels
        Returns:
            pred (List[Tensor]): len(N)
                pred[i]: size(n_pred_in_img_i, 8), post-processed restored x1,y1,x2,y2,conf,prob_class,idx_class,score from logits,
                         coordinates are in pixels, score = conf * prob_class
            logit_scale3 (Tensor): size(N, n_anchor_per_scale, img_h / 8, img_w / 8, 5 + n_class)
            logit_scale4 (Tensor): size(N, n_anchor_per_scale, img_h / 16, img_w / 16, 5 + n_class)
            logit_scale5 (Tensor): size(N, n_anchor_per_scale, img_h / 32, img_w / 32, 5 + n_class)
            loss (Tensor): size(,)
            loss_obj (Tensor): size(,)
            loss_class (Tensor): size(,)
            loss_box (Tensor): size(,)
        """
        logit_scale3, logit_scale4, logit_scale5, loss, loss_obj, loss_class, loss_box = self.forward(imgs, target)
        raw_pred = []  # inference output without NMS or post-processing

        # For scale3
        batch_size, _, n_cell_h_scale3, n_cell_w_scale3, _ = logit_scale3.shape
        if self.grid_scale3.shape[2:4] != (n_cell_h_scale3, n_cell_w_scale3):
            self.grid_scale3, self.anchor_grid_scale3 = self._make_grid(
                n_cell_h_scale3, n_cell_w_scale3, self.anchors_scale3, self.stride_scale3
            )
        xy_scale3, wh_scale3, conf_scale3, prob_class_scale3 = torch.split(
            torch.sigmoid(logit_scale3), [2, 2, 1, self.config.n_class], dim=-1
        )
        xy_scale3 = (xy_scale3 * 2 - 0.5 + self.grid_scale3) * self.stride_scale3  # xy = (2.0 * sigmoid(t_xy) - 0.5 + c_xy) * stride
        wh_scale3 = (wh_scale3 * 2) ** 2 * self.anchor_grid_scale3  # wh = (2.0 * sigmoid(t_wh)) ** 2 * p_wh
        pred_scale3 = torch.cat([xy_scale3, wh_scale3, conf_scale3, prob_class_scale3], dim=-1)
        raw_pred.append(pred_scale3.view(
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
        xy_scale4 = (xy_scale4 * 2 - 0.5 + self.grid_scale4) * self.stride_scale4  # x,y
        wh_scale4 = (wh_scale4 * 2) ** 2 * self.anchor_grid_scale4  # w,h
        pred_scale4 = torch.cat([xy_scale4, wh_scale4, conf_scale4, prob_class_scale4], dim=-1)
        raw_pred.append(pred_scale4.view(
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
        xy_scale5 = (xy_scale5 * 2 - 0.5 + self.grid_scale5) * self.stride_scale5  # x,y
        wh_scale5 = (wh_scale5 * 2) ** 2 * self.anchor_grid_scale5  # w,h
        pred_scale5 = torch.cat([xy_scale5, wh_scale5, conf_scale5, prob_class_scale5], dim=-1)
        raw_pred.append(pred_scale5.view(
            batch_size, self.config.n_anchor_per_scale * n_cell_h_scale5 * n_cell_w_scale5, 5 + self.config.n_class
        ))

        # Restored x,y,w,h,conf,prob_class from logits (coordinates are in pixels)
        raw_pred = torch.cat(raw_pred, dim=1)
        # size(N, n_raw_pred, 5 + n_class), n_raw_pred = n_anchor_per_scale * img_h / 8 * img_w / 8
        #                                                + n_anchor_per_scale * img_h / 16 * img_w / 16
        #                                                + n_anchor_per_scale * img_h / 32 * img_w / 32

        # Post-process
        pred = []  # inference output with NMS and post-processing
        img_h, img_w = imgs.shape[2:4]
        for idx_img, pred_per_img in enumerate(raw_pred):  # pred_per_img: size(n_raw_pred, 5 + n_class)
            # Score thresholding & box clipping
            score = pred_per_img[:, 4:5] * pred_per_img[:, 5:]  # size(n_raw_pred, n_class), conf * prob_class
            thresh_idx_pred, thresh_idx_class = torch.where(score > self.config.score_thresh)  # size(n_thresh_pred,)
            pred_per_img = torch.cat((
                clip_boxes_to_image(box_convert(pred_per_img[thresh_idx_pred, :4], in_fmt='cxcywh', out_fmt='xyxy'),
                                    size=(img_h, img_w)),  # size(n_thresh_pred, 4), x1,y1,x2,y2
                pred_per_img[thresh_idx_pred, 4:5],  # size(n_thresh_pred, 1), conf
                pred_per_img[thresh_idx_pred, 5 + thresh_idx_class].unsqueeze(-1),  # size(n_thresh_pred, 1), prob_class
                thresh_idx_class.unsqueeze(-1),  # size(n_thresh_pred, 1), idx_class
                score[thresh_idx_pred, thresh_idx_class].unsqueeze(-1),  # size(n_thresh_pred, 1), score
            ), dim=-1).to(torch.float32)  # size(n_thresh_pred, 8), x1,y1,x2,y2,conf,prob_class,idx_class,score
            # Clip boxes to non-pad image border
            if border is not None:
                border_per_img = border[idx_img]  # size(4,), x1,y1,x2,y2 in pixels
                pred_per_img[:, 0].clamp_(min=border_per_img[0])  # x1
                pred_per_img[:, 1].clamp_(min=border_per_img[1])  # y1
                pred_per_img[:, 2].clamp_(max=border_per_img[2])  # x2
                pred_per_img[:, 3].clamp_(max=border_per_img[3])  # y2
            # NMS
            nms_idx = batched_nms(  # don't work for BFloat16
                boxes=pred_per_img[:, :4],  # size(n_thresh_pred, 4)
                scores=pred_per_img[:, 7],  # size(n_thresh_pred,)
                idxs=pred_per_img[:, 6].to(torch.int64), # size(n_thresh_pred,)
                iou_threshold=self.config.iou_thresh
            )  # size(n_nms_pred,)
            pred_per_img = pred_per_img[nms_idx]  # size(n_nms_pred, 8)
            pred.append(pred_per_img)

        return pred, logit_scale3, logit_scale4, logit_scale5, loss, loss_obj, loss_class, loss_box


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
        grid = torch.stack((cell_x, cell_y), dim=-1).expand(shape)  # grid offset c_xy, i.e. xy_scale = 2.0 * sigmoid(t_xy) - 0.5 + c_xy
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
    target = torch.tensor([[0, 1, 0.51, 0.5, 0.5, 0.5], [0, 1, 0.51, 0.5, 0.5, 0.5]], dtype=torch.float32)
    border = torch.tensor([[0, 0, config.img_w, config.img_h], [0, 0, config.img_w, config.img_h]], dtype=torch.int64)

    logit_scale3, logit_scale4, logit_scale5, loss, _, _, _ = model(imgs, target)
    print(f"\nlogits shape:\n{logit_scale3.shape=}\n{logit_scale4.shape=}\n{logit_scale5.shape=}\n")
    if loss is not None:
        print(f"loss shape: {loss.shape} (value={loss})\n")

    pred, logit_scale3, logit_scale4, logit_scale5, loss, _, _, _ = model.generate(imgs, target, border)
    print(f"\nlogits shape:\n{len(pred)=}, {pred[0].shape=}\n{logit_scale3.shape=}\n{logit_scale4.shape=}\n{logit_scale5.shape=}\n")
    if loss is not None:
        print(f"loss shape: {loss.shape} (value={loss})\n")
