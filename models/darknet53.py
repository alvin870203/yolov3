"""
Full definition of a Darknet53 model, all of it in this single file.
Ref:
1) the official Darknet implementation:
https://github.com/pjreddie/darknet/blob/master/examples/classifier.c
https://github.com/pjreddie/darknet/blob/master/cfg/darknet53.cfg
https://github.com/pjreddie/darknet/blob/master/cfg/darknet53_448.cfg
2) unofficial pytorch implementation:
https://github.com/njustczr/cspdarknet53/blob/master/darknet53/darknet53.py
https://github.com/developer0hye/PyTorch-Darknet53/blob/master/model.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class Darknet53Config:
    img_h: int = 256
    img_w: int = 256
    n_class: int = 1000


class Darknet53Conv2d(nn.Module):
    """
    A Conv2d layer with a BarchNorm2d and a LeakyReLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        # Darknet implementation uses bias=False when batch norm is used.
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-06, momentum=0.01)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x, 0.1, inplace=True)


class Darknet53Block(nn.Module):
    """
    A Darknet53 block with 1x1 conv, 3x3 conv, and a residual connection.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1x1 = Darknet53Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = Darknet53Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        return x + identity


class Darknet53Backbone(nn.Module):
    """
    Backbone of the Darknet53 model.
    """
    def __init__(self, config: Darknet53Config) -> None:
        super().__init__()
        self.conv0 = Darknet53Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        self.conv1 = Darknet53Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.stage1_block1 = Darknet53Block(64, 64)

        self.conv2 = Darknet53Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.stage2_block1 = Darknet53Block(128, 128)
        self.stage2_block2 = Darknet53Block(128, 128)

        self.conv3 = Darknet53Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.stage3_block1 = Darknet53Block(256, 256)
        self.stage3_block2 = Darknet53Block(256, 256)
        self.stage3_block3 = Darknet53Block(256, 256)
        self.stage3_block4 = Darknet53Block(256, 256)
        self.stage3_block5 = Darknet53Block(256, 256)
        self.stage3_block6 = Darknet53Block(256, 256)
        self.stage3_block7 = Darknet53Block(256, 256)
        self.stage3_block8 = Darknet53Block(256, 256)

        self.conv4 = Darknet53Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.stage4_block1 = Darknet53Block(512, 512)
        self.stage4_block2 = Darknet53Block(512, 512)
        self.stage4_block3 = Darknet53Block(512, 512)
        self.stage4_block4 = Darknet53Block(512, 512)
        self.stage4_block5 = Darknet53Block(512, 512)
        self.stage4_block6 = Darknet53Block(512, 512)
        self.stage4_block7 = Darknet53Block(512, 512)
        self.stage4_block8 = Darknet53Block(512, 512)

        self.conv5 = Darknet53Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.stage5_block1 = Darknet53Block(1024, 1024)
        self.stage5_block2 = Darknet53Block(1024, 1024)
        self.stage5_block3 = Darknet53Block(1024, 1024)
        self.stage5_block4 = Darknet53Block(1024, 1024)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): size(N, 3, img_h, img_w)
        Returns:
            feat_stage3 (Tensor): (N, 256, img_h / 256 * 32, img_w / 256 * 32)
            feat_stage4 (Tensor): (N, 512, img_h / 256 * 16, img_w / 256 * 16)
            feat_stage5 (Tensor): (N, 1024, img_h / 256 * 8, img_w / 256 * 8)
        """
        # N x 3 x img_h x img_w
        x = self.conv0(x)
        # N x 32 x img_h x img_w

        x = self.conv1(x)
        # N x 64 x img_h / 2 x img_w / 2
        x = self.stage1_block1(x)
        # N x 64 x img_h / 2 x img_w / 2

        x = self.conv2(x)
        # N x 128 x img_h / 4 x img_w / 4
        x = self.stage2_block1(x)
        # N x 128 x img_h / 4 x img_w / 4
        x = self.stage2_block2(x)
        # N x 128 x img_h / 4 x img_w / 4

        x = self.conv3(x)
        # N x 256 x img_h / 8 x img_w / 8
        x = self.stage3_block1(x)
        # N x 256 x img_h / 8 x img_w / 8
        x = self.stage3_block2(x)
        # N x 256 x img_h / 8 x img_w / 8
        x = self.stage3_block3(x)
        # N x 256 x img_h / 8 x img_w / 8
        x = self.stage3_block4(x)
        # N x 256 x img_h / 8 x img_w / 8
        x = self.stage3_block5(x)
        # N x 256 x img_h / 8 x img_w / 8
        x = self.stage3_block6(x)
        # N x 256 x img_h / 8 x img_w / 8
        x = self.stage3_block7(x)
        # N x 256 x img_h / 8 x img_w / 8
        x = self.stage3_block8(x)
        feat_stage3 = x
        # N x 256 x img_h / 8 x img_w / 8

        x = self.conv4(x)
        # N x 512 x img_h / 16 x img_w / 16
        x = self.stage4_block1(x)
        # N x 512 x img_h / 16 x img_w / 16
        x = self.stage4_block2(x)
        # N x 512 x img_h / 16 x img_w / 16
        x = self.stage4_block3(x)
        # N x 512 x img_h / 16 x img_w / 16
        x = self.stage4_block4(x)
        # N x 512 x img_h / 16 x img_w / 16
        x = self.stage4_block5(x)
        # N x 512 x img_h / 16 x img_w / 16
        x = self.stage4_block6(x)
        # N x 512 x img_h / 16 x img_w / 16
        x = self.stage4_block7(x)
        # N x 512 x img_h / 16 x img_w / 16
        x = self.stage4_block8(x)
        feat_stage4 = x
        # N x 512 x img_h / 16 x img_w / 16

        x = self.conv5(x)
        # N x 1024 x img_h / 32 x img_w / 32
        x = self.stage5_block1(x)
        # N x 1024 x img_h / 32 x img_w / 32
        x = self.stage5_block2(x)
        # N x 1024 x img_h / 32 x img_w / 32
        x = self.stage5_block3(x)
        # N x 1024 x img_h / 32 x img_w / 32
        x = self.stage5_block4(x)
        feat_stage5 = x
        # N x 1024 x img_h / 32 x img_w / 32

        return feat_stage3, feat_stage4, feat_stage5


class Darknet53(nn.Module):
    """
    Darknet53 classification model.
    """
    def __init__(self, config: Darknet53Config) -> None:
        super().__init__()
        self.config = config

        self.backbone = Darknet53Backbone(config)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, config.n_class),
        )

        # Init all weights
        self.apply(self._init_weights)

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # TODO: zero_init_last / trunc_normal_ / head_init_scale in timm?


    def _compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the cross entropy loss.
        Args:
            logits (Tensor): size(N, n_class)
            targets (Tensor): size(N,)
        Returns:
            loss (Tensor): size(,)
        """
        return F.cross_entropy(logits, targets)


    def forward(self, imgs: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            imgs (Tensor): size(N, 3, img_h, img_w)
            targets (Tensor): size(N, n_class)
        Returns:
            logits (Tensor): size(N,)
            loss (Tensor): size(,)
        """
        device = imgs.device

        # Forward the Darknet53 model itself
        # N x 3 x img_h x img_w
        feat_stage3, feat_stage4, feat_stage5 = self.backbone(imgs)
        # N x 256 x img_h / 8 x img_w / 8; N x 512 x img_h / 16 x img_w / 16; N x 1024 x img_h / 32 x img_w / 32
        logits = self.head(feat_stage5)
        # N x n_class

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            loss = self._compute_loss(logits, targets)
        else:
            loss = None

        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError("FUTURE: init from pretrained model")


    def configure_optimizers(self, optimizer_type, learning_rate, betas, weight_decay, device_type, use_fused):
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups. any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls decay, all biases and norms don't.
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
            optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, momentum=betas[0])
            print(f"using SGD")
        else:
            raise ValueError(f"unrecognized optimizer_type: {optimizer_type}")

        return optimizer


    def estimate_tops(self):
        """
        Estimate the number of TOPS and parameters in the model.
        """
        raise NotImplementedError("FUTURE: estimate TOPS for Darknet53 model")


    @torch.inference_mode()
    def generate(self, imgs, top_k=None):
        """
        Predict on test imgs and return the top_k predictions.
        """
        # Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        self.eval()
        raise NotImplementedError("FUTURE: generate for Darknet53 model")
        self.train()



if __name__ == '__main__':
    # Test the model by `python -m models.darknet53` from the workspace directory
    config = Darknet53Config()
    # config = Darknet53Config(img_h=256, img_w=256)
    # config = Darknet53Config(img_h=448, img_w=448)
    model = Darknet53(config)
    print(model)
    print(f"num params: {model.get_num_params():,}")

    imgs = torch.randn(2, 3, config.img_h, config.img_w)
    targets = torch.randint(0, config.n_class, (2,))
    logits, loss = model(imgs, targets)
    print(f"logits shape: {logits.shape}")
    if loss is not None:
        print(f"loss shape: {loss.shape}")
