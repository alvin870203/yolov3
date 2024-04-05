# Config for training Darknet53 model on no-aug ImageNet2012 dataset for image classification
# as the upper-bound training accuracy (overfit) baseline of the backbone for YOLOv3
import time

# Task related
task_name = 'classify'
init_from = 'scratch'

# Data related
dataset_name = 'imagenet2012'
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
# 256 batch_size * 1 grad_accum = 256 imgs/iter
# imagenet2012 train set has 1,281,167 imgs, so 1 epoch ~= 5,005 iters
gradient_accumulation_steps = 1
batch_size = 256  # filled up the gpu memory on my machine
max_iters = 500500  # 100 epochs, finish in 48 hr on my machine

# Optimizer related
optimizer_type = 'sgd'
learning_rate = 0.1
beta1 = 0.9  # momentum
#beta2 = 0.999  # not used in sgd
weight_decay = 1e-4
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = 'step'  # whether to decay the learning rate
warmup_iters = 0  # no warmup
lr_decay_iters = 150150  # decay every 30 epochs
min_lr = 0.1  # decay 1/10 every lr_decay_iters
use_fused = True  # somehow use_fused=True is incompatible to compile=True in this model

# Eval related
# imagenet2012 val set has 50,000 imgs, so 1 epoch ~= 196 iters
eval_interval = 5005  # keep frequent if we'll overfit
eval_iters = 196  # use entire val to get good estimate

# Log related
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
out_dir = f'out/darknet53_imagenet2012/{timestamp}'
wandb_log = True
wandb_project = 'imagenet2012'
wandb_run_name = f'darknet53_{timestamp}'
log_interval = 200  # don't print too often
always_save_checkpoint = False  # only save when val improves if we expect overfit

# System related
compile = False  # somehow use_fused=True is incompatible to compile=True in this model
n_worker = 8
