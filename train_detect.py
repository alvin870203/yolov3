"""
Training script for a detector.
To run, example:
$ python train_detect.py config/train_yolov3_voc.py --n_worker=1
"""


import os
import time
import math
import pickle
from contextlib import nullcontext
import random
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
import torch.multiprocessing
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.ops import box_convert
from torchvision.datasets import wrap_dataset_for_transforms_v2
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision

from evaluator import DetEvaluator
from models.yolov3 import Yolov3Config, Yolov3


# -----------------------------------------------------------------------------
# Default config values
# Task related
task_name = 'detect'
eval_only = False  # if True, script exits right after the first eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'backbone' or 'pretrained'
from_ckpt = ''  # only used when init_from='resume' or 'backbone' or 'pretrained'
# Data related
dataset_name = 'voc'  # 'voc' or 'nano_voc' or 'blank_voc' or 'nano_blank_voc'
img_h = 416  # should be multiple of max stride
img_w = 416  # should be multiple of max stride
n_class = 20  # 20 for voc, 80 for coco
# Transform related
aug_type = 'default'  # 'default' or sannapersson'
letterbox = True
fill = (123.0, 117.0, 104.0)
color_p = 0.4
brightness = 0.4
contrast = 0.4
saturation = 0.7
hue = 0.015
blur_p = 0.1
blur_size_min = 3
blur_size_max = 7
blur_sigma_min = 0.1
blur_sigma_max = 2.0
autocontrast_p = 0.1
posterize_p = 0.1
posterize_bits = 4
grayscale_p = 0.1
channelshuffle_p = 0.05
perspective_p = 0.4
perspective = 0.0
translate = 0.1
scale = 0.75
shear_p = 0.4
shear = 0.0  # unit: deg
rotate_p = 0.4
degrees = 0.0  # unit: deg
crop_scale = 0.8
ratio_min = 0.5
ratio_max = 2.0
flip_p = 0.5
min_size = 1.0  # filter out too small boxes in augmented training data
imgs_mean = (0.485, 0.456, 0.406)
imgs_std = (0.229, 0.224, 0.225)
multiscale_h = (320, 352, 384, 416, 448, 480, 512, 544, 576, 608)  # should be multiple of max stride, (img_h,) to disable
multiscale_w = (320, 352, 384, 416, 448, 480, 512, 544, 576, 608)  # should be multiple of max stride, (img_w,) to disable
# Model related
model_name = 'yolov3'
n_scale = 3
n_anchor_per_scale = 3
anchors = (  # size(n_scale, n_anchor_per_scale, 2)
    ((10, 13), (16, 30), (33, 23)),  # scale3
    ((30, 61), (62, 45), (59, 119)),  # scale4
    ((116, 90), (156, 198), (373, 326)),  # scale5
)  # w,h in pixels of a 416x416 image. IMPORTANT: from scale3 to scale5, order-aware!
# Loss related
match_thresh = 4.0  # iou or wh ratio threshold to match a target to an anchor when calculating loss
rescore = 1.0  # 0.0~1.0, rescore ratio; if 0.0, use 1 as objectness target; if 1.0, use iou as objectness target
smooth = 0.0  # 0.0~1.0, smooth ratio for class BCE loss; 0.0 for no smoothing; 0.1 is a common choice
pos_weight_class = 1.0  # weight for class BCE loss of positive examples, as if the positive examples are duplicated
pos_weight_obj = 1.0  # weight for obj BCE loss of positive examples, as if the positive examples are duplicated
balance = (4.0, 1.0, 0.4)  # balance weights for scale3~5 loss_obj.
lambda_box = 0.05
lambda_obj = 1.0
lambda_class = 0.5
# Train related
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
batch_size = 2  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_iters = 100000  # total number of training iterations
# Optimizer related
optimizer_type = 'adamw'  # 'adamw' or 'adam' or 'sgd'
learning_rate = 1e-3  # max learning rate
beta1 = 0.9  # beta1 for adamw, momentum for sgd
beta2 = 0.999
weight_decay = 1e-2
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = 'cosine'  # whether to decay the learning rate, which type of lr scheduler. False, 'cosine', 'step'
warmup_iters = 5000  # how many steps to warm up for
lr_decay_iters = 100000  # should be ~= max_iters; this is milestones if decay_lr='step'
min_lr = 1e-4  # minimum learning rate, should be ~= learning_rate/10; this is gamma if decay_lr='step'
use_fused = True  # whether to use fused optimizer kernel
# Eval related
eval_interval = 100  # keep frequent if we'll overfit
eval_iters = 200  # use more iterations to get good estimate
score_thresh = 0.001  # threshold for (objectness score * class probability) when filtering inference results
iou_thresh = 0.6  # NMS iou threshold when filtering inference results, 0.5 for mAP50, 0.6 for mAP
use_torchmetrics = False  # whether to use cocoeval for detailed but slower metric computation
# Log related
timestamp = '00000000-000000'
out_dir = f'out/yolov3_voc/{timestamp}'
wandb_log = False  # disabled by default
wandb_project = 'voc'
wandb_run_name = f'yolov3_{timestamp}'
log_interval = 50  # don't print too often
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
# System related
device = 'cuda'  # example: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 to compile the model to be faster
n_worker = 0
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, tuple))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


# Various inits, derived attributes, I/O setup
imgs_per_iter = gradient_accumulation_steps * batch_size
print(f"imgs_per_iter will be: {imgs_per_iter}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# torch.multiprocessing.set_sharing_strategy('file_system')  # if too many open files error, try this, but check if no slowing down


# Dataloader
data_dir = os.path.join('data', dataset_name.strip('nano_').strip('blank_'))
if dataset_name == 'voc' or dataset_name == 'nano_voc':
    from dataloaders.voc import VocConfig, voc_collate_fn, VocTrainDataLoader, VocValDataLoader
    dataloader_args = dict(
        img_h=img_h, img_w=img_w, aug_type=aug_type, letterbox=letterbox, fill=fill,
        color_p=color_p, brightness=brightness, contrast=contrast, saturation=saturation, hue=hue,
        blur_p=blur_p, blur_size_min=blur_size_min, blur_size_max=blur_size_max,
        blur_sigma_min=blur_sigma_min, blur_sigma_max=blur_sigma_max, autocontrast_p=autocontrast_p,
        posterize_p=posterize_p, posterize_bits=posterize_bits, grayscale_p=grayscale_p,
        channelshuffle_p=channelshuffle_p, perspective_p=perspective_p, perspective=perspective,
        translate=translate, scale=scale, shear_p=shear_p, shear=shear, rotate_p=rotate_p, degrees=degrees,
        crop_scale=crop_scale, ratio_min=ratio_min, ratio_max=ratio_max, flip_p=flip_p, min_size=min_size,
        imgs_mean=imgs_mean, imgs_std=imgs_std,
    )
    dataloader_config = VocConfig(**dataloader_args)
    dataloaders = {
        'train': VocTrainDataLoader(dataloader_config, data_dir, batch_size=batch_size, num_workers=n_worker,
                                    collate_fn=voc_collate_fn, nano=dataset_name.startswith('nano_')),
        'val': VocValDataLoader(dataloader_config, data_dir, batch_size=batch_size, num_workers=n_worker,
                                collate_fn=voc_collate_fn, shuffle=True,  # shuffle=True since only eval partial data
                                nano=dataset_name.startswith('nano_'))
    }
elif dataset_name == 'blank_voc' or dataset_name == 'nano_blank_voc':
    from dataloaders.voc import BlankVocConfig, voc_collate_fn, BlankVocTrainDataLoader, BlankVocValDataLoader
    dataloader_args = dict(img_h=img_h, img_w=img_w, letterbox=letterbox, fill=fill, min_size=min_size,
                           imgs_mean=imgs_mean, imgs_std=imgs_std)
    dataloader_config = BlankVocConfig(**dataloader_args)
    dataloaders = {
        'train': BlankVocTrainDataLoader(dataloader_config, data_dir, batch_size=batch_size, num_workers=n_worker,
                                           collate_fn=voc_collate_fn, nano=dataset_name.startswith('nano_')),
        'val': BlankVocValDataLoader(dataloader_config, data_dir, batch_size=batch_size, num_workers=n_worker,
                                     collate_fn=voc_collate_fn, shuffle=True,  # shuffle=True since only eval partial data
                                     nano=dataset_name.startswith('nano_'))
    }
else:
    raise ValueError(f"dataset_name: {dataset_name} not supported")
print(f"train dataset: {len(dataloaders['train'].dataset)} samples, {len(dataloaders['train'])} batches")
print(f"val dataset: {len(dataloaders['val'].dataset)} samples, {len(dataloaders['val'])} batches")

# Init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
multiscale_img_h, multiscale_img_w = img_h, img_w

# Batch getter
dataiters = {'train': iter(dataloaders['train']), 'val': iter(dataloaders['val'])}
def get_batch(split):
    try:
        X, Y, BORDER = next(dataiters[split])
    except StopIteration:
        dataiters[split] = iter(dataloaders[split])
        X, Y, BORDER = next(dataiters[split])
    if device_type == 'cuda':
        # X, Y, BORDER is pinned in dataloader, which allows us to move them to GPU asynchronously (non_blocking=True)
        X, Y, BORDER = X.to(device, non_blocking=True), Y.to(device, non_blocking=True), BORDER.to(device, non_blocking=True)
    else:
        X, Y, BORDER = X.to(device), Y.to(device), BORDER.to(device)
    return X, Y, BORDER


# Model init
model_args = dict(
    img_h=img_h, img_w=img_w, n_class=n_class, n_scale=n_scale, n_anchor_per_scale=n_anchor_per_scale, anchors=anchors,
    match_thresh=match_thresh, rescore=rescore, smooth=smooth,
    pos_weight_class=pos_weight_class, pos_weight_obj=pos_weight_obj, balance=balance,
    lambda_box=lambda_box, lambda_obj=lambda_obj, lambda_class=lambda_class,
    score_thresh=score_thresh, iou_thresh=iou_thresh,
)  # start with model_args from command line

if init_from == 'scratch':
    # Init a new model from scratch
    print(f"Initializing a new {model_name} model from scratch")
    model_config = Yolov3Config(**model_args)
    model = Yolov3(model_config)
elif init_from == 'resume':
    print(f"Resuming training {model_name} from {from_ckpt}")
    # Resume training from a checkpoint
    checkpoint = torch.load(from_ckpt, map_location='cpu')  # load to CPU first to avoid GPU OOM
    checkpoint_model_args = checkpoint['model_args']
    assert model_name == checkpoint['config']['model_name'], "model_name mismatch"
    assert dataset_name == checkpoint['config']['dataset_name'], "dataset_name mismatch"
    # Force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['img_h', 'img_w', 'n_class', 'n_scale', 'n_anchor_per_scale']:
        model_args[k] = checkpoint_model_args[k]
    # Create the model
    model_config = Yolov3Config(**model_args)
    model = Yolov3(model_config)
    state_dict = checkpoint['model']
    # Fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    multiscale_img_h, multiscale_img_w = checkpoint['multiscale_imgsz']
elif init_from == 'backbone':
    print(f"Initializing a {model_name} model with pretrained backbone weights: {from_ckpt}")
    # Init a new model with pretrained backbone weights
    checkpoint = torch.load(from_ckpt, map_location='cpu')
    model_config = Yolov3Config(**model_args)
    model = Yolov3(model_config)
    state_dict = checkpoint['model']
    wanted_prefix = 'backbone.'
    for k,v in list(state_dict.items()):
        if not k.startswith(wanted_prefix):
            state_dict.pop(k)
        else:
            state_dict[k[len(wanted_prefix):]] = state_dict.pop(k)
    model.backbone.load_state_dict(state_dict)
elif init_from == 'pretrained':
    print(f"Initializing a {model_name} model with entire pretrained weights: {from_ckpt}")
    # Init a new model with entire pretrained weights
    checkpoint = torch.load(from_ckpt, map_location='cpu')
    model_config = Yolov3Config(**model_args)
    model = Yolov3(model_config)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
else:
    raise ValueError(f"Invalid init_from: {init_from}")

model = model.to(device)


# Initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))


# Optimizer
optimizer = model.configure_optimizers(optimizer_type, learning_rate, (beta1, beta2), weight_decay, device_type, use_fused)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory


# Compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0


# Helps estimate an arbitrarily accurate loss over either split using many batches
# Not accurate since losses were averaged at each iteration (exact would have been a sum),
# then averaged altogether again at the end, but the metrics are accurate.
@torch.inference_mode()
def estimate_loss():
    out_losses, out_map50, out_losses_obj, out_losses_class, out_losses_box = {}, {}, {}, {}, {}
    model.eval()
    for split in ['train', 'val']:
        n_eval_steps = int(eval_iters * gradient_accumulation_steps)
        losses, losses_obj, losses_class, losses_box = \
            torch.zeros(n_eval_steps), torch.zeros(n_eval_steps), torch.zeros(n_eval_steps), torch.zeros(n_eval_steps)
        if use_torchmetrics:
            metric = MeanAveragePrecision(iou_type='bbox')
            metric.warn_on_many_detections = False
        else:
            metric = DetEvaluator()
        for idx_step in range(n_eval_steps):
            X, Y, BORDER = get_batch(split)
            eval_batch_size, _, eval_img_h, eval_img_w = X.shape
            with ctx:
                pred, _, _, _, loss, loss_obj, loss_class, loss_box = model.generate(X, Y, BORDER)
            losses[idx_step], losses_obj[idx_step], losses_class[idx_step], losses_box[idx_step] = \
                loss.item(), loss_obj.item(), loss_class.item(), loss_box.item()
            pred_for_eval = [
                dict(boxes=pred_per_img[:, :4], scores=pred_per_img[:, 7], labels=pred_per_img[:, 6].to(torch.int64))
                for pred_per_img in pred
            ]
            Y_box = Y[:, 2:] * torch.tensor([eval_img_w, eval_img_h, eval_img_w, eval_img_h], device=device)  # to unit of pixels
            Y_box = box_convert(Y_box, in_fmt='cxcywh', out_fmt='xyxy')  # to x1y1x2y2 format
            Y_class = Y[:, 1].to(torch.int64)
            mask_target = [Y[:, 0] == idx_img for idx_img in range(eval_batch_size)]  # list of masks
            target_for_eval = [dict(boxes=Y_box[mask_target_per_img], labels=Y_class[mask_target_per_img])
                               for mask_target_per_img in mask_target]
            metric.update(pred_for_eval, target_for_eval)
        map50 = metric.compute()['map_50'] * 100
        out_map50[split] = map50.item() if isinstance(map50, torch.Tensor) else map50
        out_losses[split], out_losses_obj[split], out_losses_class[split], out_losses_box[split] = \
            losses.mean(), losses_obj.mean(), losses_class.mean(), losses_box.mean()
    model.train()
    return out_losses, out_losses_obj, out_losses_class, out_losses_box, out_map50


# Learning rate decay scheduler (cosine with warmup)
if decay_lr == 'cosine':
    def get_lr(it):
        # 1) Linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) If it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) In between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)
elif decay_lr == 'step':
    def get_lr(it):
        # 1) Linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) After warmup, use step decay with min_lr as gamma
        # lr_decay_iters is a tuple of milestones
        return learning_rate * (min_lr ** sum(it >= milestone for milestone in lr_decay_iters))
elif decay_lr == False:
    pass
else:
    raise ValueError(f"Invalid decay_lr: {decay_lr}")


# Logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# Training loop
X, Y, BORDER = get_batch('train')  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
pbar = tqdm(total=max_iters, initial=iter_num, dynamic_ncols=True)

while True:

    # Determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses, losses_obj, losses_class, losses_box, map50 = estimate_loss()
        tqdm.write(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, train map50 {map50['train']:.4f}, val map50 {map50['val']:.4f}")
        if wandb_log:
            wandb.log({"iter": iter_num, "lr": lr,
                       "train/loss": losses['train'], "val/loss": losses['val'],
                       "train/loss_obj": losses_obj['train'], "val/loss_obj": losses_obj['val'],
                       "train/loss_class": losses_class['train'], "val/loss_class": losses_class['val'],
                       "train/loss_box": losses_box['train'], "val/loss_box": losses_box['val'],
                       "train/map50": map50['train'], "val/map50": map50['val']})
        checkpoint = {'config': config, 'model_args': model_args, 'rng_state': torch.get_rng_state(),
                      'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'iter_num': iter_num,
                      'best_val_loss': best_val_loss, 'multiscale_imgsz': (multiscale_img_h, multiscale_img_w)}
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            checkpoint['best_val_loss'] = best_val_loss
            if iter_num > 0:
                tqdm.write(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))  # TODO: save top k checkpoints
        if iter_num + eval_interval > max_iters:  # last eval
            checkpoint['best_val_loss'] = losses['val']
            tqdm.write(f"saving last checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt_last.pt'))

    if iter_num == 0 and eval_only:
        break

    # Multi-scale training, select new img size every 10 iterations
    if iter_num > 0 and iter_num % 10 == 0:
        idx_multiscale = torch.randint(0, len(multiscale_h), (1,)).item()
        multiscale_img_h, multiscale_img_w = multiscale_h[idx_multiscale], multiscale_w[idx_multiscale]
    X = nn.functional.interpolate(X, size=(multiscale_img_h, multiscale_img_w), mode='bilinear', align_corners=False)

    # Forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logit_scale3, logit_scale4, logit_scale5, loss, loss_obj, loss_class, loss_box = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

        # Immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, BORDER = get_batch('train')

        # Backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # Clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    # Flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # TODO: auto adjust labmda_box, lambda_obj, lambda_class based on loss_obj, loss_class, loss_box

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # Get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        tqdm.write(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1
    pbar.update(1)

    # Termination conditions
    if iter_num > max_iters:
        break
