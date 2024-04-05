# Config for finetuning Darknet53 to 448x448 model on tv-aug ImageNet2012 dataset for image classification as the backbone for YOLOv3
import time

# Task related
task_name = 'classify'
init_from = 'pretrained'
from_ckpt = 'saved/darknet53_imagenet2012/20240401-084004/ckpt_last.pt'

# Data related
dataset_name = 'imagenet2012'
img_h = 448
img_w = 448
n_class = 1000

# Transform related
imgs_mean = (0.0, 0.0, 0.0)
imgs_std = (1.0, 1.0, 1.0)
aug_type = 'tv-aug'
scale_min = 0.08
scale_max = 1.0
ratio_min = 3.0 / 4.0
ratio_max = 4.0 / 3.0
flip_p = 0.5
fill = (123.0, 117.0, 104.0)

# Model related
model_name = 'darknet53'

# Train related
# the number of examples per iter:
# 64 batch_size * 2 grad_accum = 128 imgs/iter
# imagenet2012 train set has 1,281,167 imgs, so 1 epoch ~= 10,010 iters
gradient_accumulation_steps = 2
batch_size = 64  # filled up the gpu memory on my machine
max_iters = 200200  # 20 epochs, finish in TODO hr on my machine

# Optimizer related
# SGD with step lr decay
optimizer_type = 'sgd'
learning_rate = 0.001
beta1 = 0.9  # momentum
#beta2 = 0.999  # not used in sgd
weight_decay = 1e-4
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = 'cosine'  # whether to decay the learning rate
warmup_iters = 0  # no warmup
lr_decay_iters = 200200  # should be ~= max_iters
min_lr = 0  # minimum learning rate, should be ~= learning_rate/10, but set to 0 as pytorch default
use_fused = True  # somehow use_fused=True is incompatible to compile=True in this model

# Eval related
# imagenet2012 val set has 50,000 imgs, so 1 epoch ~= 391 iters
eval_interval = 10010  # keep frequent if we'll overfit
eval_iters = 391  # use entire val to get good estimate

# Log related
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
out_dir = f'out/darknet53-448_imagenet2012/{timestamp}'
wandb_log = True
wandb_project = 'imagenet2012'
wandb_run_name = f'darknet53-448_{timestamp}'
log_interval = 200  # don't print too often
always_save_checkpoint = False  # only save when val improves if we expect overfit

# System related
compile = False  # somehow use_fused=True is incompatible to compile=True in this model
n_worker = 8
