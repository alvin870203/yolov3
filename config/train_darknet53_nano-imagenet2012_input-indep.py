# Config for training Darknet53 model on blank no-aug nano ImageNet2012 dataset for image classification debug
import time

# Task related
task_name = 'classify'
init_from = 'scratch'

# Data related
dataset_name = 'nano_blank_imagenet2012'
img_h = 256
img_w = 256
n_class = 1000

# Transform related
# No augmentation
imgs_mean = (0.0, 0.0, 0.0)
imgs_std = (1.0, 1.0, 1.0)
scale_min = 1.0
scale_max = 1.0
ratio_min = 1.0
ratio_max = 1.0
perspective = 0.0
degrees = 0.0
translate = 0.0
shear = 0.0
brightness = 0.0
contrast = 0.0
saturation = 0.0
hue = 0.0
flip_p = 0.0
letterbox = True
fill = (123.0, 117.0, 104.0)

# Model related
model_name = 'darknet53'

# Train related
# the number of examples per iter:
# 2 batch_size * 1 grad_accum = 2 imgs/iter
# nano imagenet2012 train set has 2 imgs, so 1 epoch ~= 1 iters
gradient_accumulation_steps = 1
batch_size = 2  # entire dataset
max_iters = 12800

# Optimizer related
optimizer_type = 'adamw'
learning_rate = 3e-4  # smaller lr to overfit stably
beta1 = 0.9
beta2 = 0.999
weight_decay = 5e-4
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = False  # whether to decay the learning rate
#warmup_iters = 5  # warmup 5 epochs
#lr_decay_iters = 12800  # should be ~= max_iters
#min_lr = 3e-5  # minimum learning rate, should be ~= learning_rate/10
use_fused = True  # somehow use_fused=True is incompatible to compile=True in this model

# Eval related
# nano imagenet2012 val set has 2 imgs, so 1 epoch ~= 1 iters
eval_interval = 2  # keep frequent if we'll overfit
eval_iters = 1  # use entire val to get good estimate

# Log related
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
out_dir = f'out/darknet53_nano-imagenet2012/{timestamp}'
wandb_log = True
wandb_project = 'nano-imagenet2012'
wandb_run_name = f'darknet53_{timestamp}'
log_interval = 200  # don't print too often
always_save_checkpoint = False  # only save when val improves if we expect overfit

# System related
compile = False  # somehow use_fused=True is incompatible to compile=True in this model
n_worker = 8
