"""
Training script for a classifier.
To run, example:
$ python train_classify.py config/train_darknet53_imagenet2012.py --n_worker=1
"""


import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torchvision
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Default config values
# Task related
task_name = 'classify'
eval_only = False  # if True, script exits right after the first eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'backbone' or 'pretrained'
from_ckpt = ''  # only used when init_from='resume' or 'backbone' or 'pretrained'
# Data related
dataset_name = 'imagenet2012'  # 'nano_blank_imagenet2012', 'blank_imagenet2012', 'nano_imagenet2012', or 'imagenet2012'
img_h = 256
img_w = 256
n_class = 1000
# Transform related
imgs_mean = (0.485, 0.456, 0.406)
imgs_std = (0.229, 0.224, 0.225)
aug_type = 'simple-aug'  # 'simple-aug', 'tv-aug', or 'voc-aug'
scale_min = 0.08  # 0.08 for random resized crop, 0.5 for affine transform
scale_max = 1.0  # 1.0 for random resized crop, 1.5 for affine transform
ratio_min = 3.0 / 4.0  # 3.0 / 4.0 for random resized crop, 0.5 for affine transform
ratio_max = 4.0 / 3.0  # 4.0 / 3.0 for random resized crop, 2.0 for affine transform
perspective = 0.1
degrees = 0.5
translate = 0.25
shear = 0.5
brightness = 0.4
contrast = 0.4
saturation = 0.7
hue = 0.015
flip_p = 0.5
letterbox = True
fill = (123.0, 117.0, 104.0)
# Model related
model_name = 'darknet53'
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
lr_decay_iters = 100000  # should be ~= max_iters; this is step_size if decay_lr='step'
min_lr = 1e-4  # minimum learning rate, should be ~= learning_rate/10; this is gamma if decay_lr='step'
use_fused = True  # whether to use fused optimizer kernel
# Eval related
eval_interval = 100  # keep frequent if we'll overfit
eval_iters = 200  # use more iterations to get good estimate
# Log related
timestamp = '00000000-000000'
out_dir = f'out/darknet53_imagenet2012/{timestamp}'
wandb_log = False  # disabled by default
wandb_project = 'imagenet2012'
wandb_run_name = f'darknet53_{timestamp}'
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
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# Dataloader
data_dir = os.path.join('data', dataset_name.strip('nano_').strip('blank_'))
if dataset_name == 'imagenet2012' or dataset_name == 'nano_imagenet2012':
    from dataloaders.imagenet2012 import ImageNetConfig, ImageNetTrainDataLoader, ImageNetValDataLoader
    dataloader_args = dict(
        img_h=img_h,
        img_w=img_w,
        imgs_mean=imgs_mean,
        imgs_std=imgs_std,
        aug_type=aug_type,
        scale_min=scale_min,
        scale_max=scale_max,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        perspective=perspective,
        degrees=degrees,
        translate=translate,
        shear=shear,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        flip_p=flip_p,
        letterbox=letterbox,
        fill=fill,
    )
    dataloader_config = ImageNetConfig(**dataloader_args)
    dataloader_train = ImageNetTrainDataLoader(dataloader_config, data_dir, batch_size=batch_size, num_workers=n_worker,
                                               nano=dataset_name.startswith('nano_'))
    dataloader_val = ImageNetValDataLoader(dataloader_config, data_dir, batch_size=batch_size, num_workers=n_worker,
                                           shuffle=True,  # shuffle=True since only eval partial data
                                           nano=dataset_name.startswith('nano_'))
elif dataset_name == 'blank_imagenet2012' or dataset_name == 'nano_blank_imagenet2012':
    from dataloaders.imagenet2012 import BlankImageNetConfig, BlankImageNetTrainDataLoader, BlankImageNetValDataLoader
    dataloader_args = dict(
        img_h=img_h,
        img_w=img_w,
        imgs_mean=imgs_mean,
        imgs_std=imgs_std,
        fill=fill,
    )
    dataloader_config = BlankImageNetConfig(**dataloader_args)
    dataloader_train = BlankImageNetTrainDataLoader(dataloader_config, data_dir, batch_size=batch_size, num_workers=n_worker,
                                                    nano=dataset_name.startswith('nano_'))
    dataloader_val = BlankImageNetValDataLoader(dataloader_config, data_dir, batch_size=batch_size, num_workers=n_worker,
                                                shuffle=True,  # shuffle=True since only eval partial data
                                                nano=dataset_name.startswith('nano_'))
else:
    raise ValueError(f"dataset_name: {dataset_name} not supported")

class BatchGetter:  # for looping through dataloaders is still a bit faster and less gpu memory than this
    assert len(dataloader_train) >= eval_iters, f"Not enough batches in train loader for eval."
    assert len(dataloader_val) >= eval_iters, f"Not enough batches in val loader for eval."
    dataiter = {'train': iter(dataloader_train), 'val': iter(dataloader_val)}

    @classmethod
    def get_batch(cls, split):
        try:
            X, Y = next(cls.dataiter[split])
        except StopIteration:
            cls.dataiter[split] = iter(dataloader_train) if split == 'train' else iter(dataloader_val)
            X, Y = next(cls.dataiter[split])

        if device_type == 'cuda':
            # X, Y is pinned in dataloader, which allows us to move them to GPU asynchronously (non_blocking=True)
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
        else:
            X, Y = X.to(device), Y.to(device)

        return X, Y


# Init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9


# Model init
if init_from == 'scratch':
    # Init a new model from scratch
    print(f"Initializing a new {model_name} model from scratch")
elif init_from == 'resume':
    print(f"Resuming training {model_name} from {from_ckpt}")
    # Resume training from a checkpoint
    checkpoint = torch.load(from_ckpt, map_location='cpu')  # load to CPU first to avoid GPU OOM
    torch.set_rng_state(checkpoint['rng_state'].to('cpu'))
    checkpoint_model_args = checkpoint['model_args']
    assert model_name == checkpoint['config']['model_name'], "model_name mismatch"
    assert dataset_name == checkpoint['config']['dataset_name'], "dataset_name mismatch"
elif init_from == 'backbone':
    print(f"Initializing a {model_name} model with pretrained backbone weights: {from_ckpt}")
    # Init a new model with pretrained backbone weights
    checkpoint = torch.load(from_ckpt, map_location='cpu')
elif init_from == 'pretrained':
    print(f"Initializing a {model_name} model with entire pretrained weights: {from_ckpt}")
    # Init a new model with entire pretrained weights
    checkpoint = torch.load(from_ckpt, map_location='cpu')
    torch.set_rng_state(checkpoint['rng_state'].to('cpu'))
else:
    raise ValueError(f"Invalid init_from: {init_from}")

if model_name == 'darknet53':
    from models.darknet53 import Darknet53Config, Darknet53
    model_args = dict(
        img_h=img_h,
        img_w=img_w,
        n_class=n_class
    )  # start with model_args from command line
    if init_from == 'resume':
        # Force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['img_h', 'img_w', 'n_class']:
            model_args[k] = checkpoint_model_args[k]
    # Create the model
    model_config = Darknet53Config(**model_args)
    model = Darknet53(model_config)
else:
    raise ValueError(f"model_name: {model_name} not supported")

if init_from == 'resume':
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
elif init_from == 'backbone':
    state_dict = checkpoint['model']
    wanted_prefix = 'backbone.'
    for k,v in list(state_dict.items()):
        if not k.startswith(wanted_prefix):
            state_dict.pop(k)
        else:
            state_dict[k[len(wanted_prefix):]] = state_dict.pop(k)
    model.backbone.load_state_dict(state_dict)
elif init_from == 'pretrained':
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
elif init_from == 'scratch':
    pass
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
    out_losses = {}
    out_acc1 = {}
    out_acc5 = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters * gradient_accumulation_steps)
        acc1, acc5 = 0.0, 0.0
        n_seen = 0
        for k in range(eval_iters * gradient_accumulation_steps):
            X, Y = BatchGetter.get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
            _, pred = logits.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(Y.view(1, -1).expand_as(pred))
            n_seen += X.size(0)
            acc1 += correct[:1].reshape(-1).float().sum(0).item()
            acc5 += correct[:5].reshape(-1).float().sum(0).item()
        out_losses[split] = losses.mean()
        out_acc1[split] = 100 * acc1 / n_seen
        out_acc5[split] = 100 * acc5 / n_seen
    model.train()
    return out_losses, out_acc1, out_acc5


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
        return learning_rate * (min_lr ** ((it - warmup_iters) // lr_decay_iters))
elif decay_lr == False:
    pass
else:
    raise ValueError(f"Invalid decay_lr: {decay_lr}")


# Logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# Training loop
X, Y = BatchGetter.get_batch('train')  # fetch the very first batch
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
        losses, acc1, acc5 = estimate_loss()
        tqdm.write(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val top1 acc {acc1['val']:.2f}%, val top5 acc {acc5['val']:.2f}%")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "train/acc1": acc1['train'],
                "train/acc5": acc5['train'],
                "val/loss": losses['val'],
                "val/acc1": acc1['val'],
                "val/acc5": acc5['val'],
                "lr": lr
            })

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': config,
            'rng_state': torch.get_rng_state()
        }
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

    # Forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

        # Immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = BatchGetter.get_batch('train')

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
