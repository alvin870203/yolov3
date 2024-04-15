# Config for training YOLOv3 model on no-aug Pascal VOC 2007&2012 VOC dataset for object detection
# as the upper-bound training accuracy (overfit) baseline
# Train on VOC 2007 trainval and 2012 trainval, and evaluate on nano VOC 2007 test
import time

# Task related
task_name = 'detect'
init_from = 'backbone'
from_ckpt = 'saved/darknet53-448_imagenet2012/20240404-112658/ckpt_last.pt'

# Data related
dataset_name = 'voc'
img_h = 416
img_w = 416
n_class = 20

# Transform related
# No augmentation
aug_type = 'default'
letterbox = True
fill = (123.0, 117.0, 104.0)
color_p = 0.0
brightness = 0.0
contrast = 0.0
saturation = 0.0
hue = 0.0
blur_p = 0.0
blur_size_min = 3
blur_size_max = 7
blur_sigma_min = 0.1
blur_sigma_max = 2.0
autocontrast_p = 0.0
posterize_p = 0.0
posterize_bits = 4
grayscale_p = 0.0
channelshuffle_p = 0.0
perspective_p = 0.0
perspective = 0.0
translate = 0.0
scale = 0.0
shear_p = 0.0
shear = 0.0
rotate_p = 0.0
degrees = 0.0
crop_scale = 1.0
ratio_min = 1.0
ratio_max = 1.0
flip_p = 0.0
min_size = 1.0
imgs_mean = (0.0, 0.0, 0.0)
imgs_std = (1.0, 1.0, 1.0)
multiscale_h = (416,)
multiscale_w = (416,)

# Model related
model_name = 'yolov3'
n_scale = 3
n_anchor_per_scale = 3
anchors = (((10, 13), (16, 30), (33, 23)), ((30, 61), (62, 45), (59, 119)), ((116, 90), (156, 198), (373, 326)))

# Loss related
match_thresh = 4.0
rescore = 1.0
smooth = 0.0
pos_weight_class = 1.0
pos_weight_obj = 1.0
balance = (4.0, 1.0, 0.4)
lambda_box = 0.05
lambda_obj = 1.0
lambda_class = 0.5

# Train related
# the number of examples per iter:
# 64 batch_size * 1 grad_accum = 64 imgs/iter
# voc train set has 16,551 imgs, so 1 epoch ~= 259 iters
gradient_accumulation_steps = 1
batch_size = 64  # filled up the gpu memory on my machine
max_iters = 77700  # 300 epochs, finish in TODO hr on my machine

# Optimizer related
optimizer_type = 'sgd'
learning_rate = 1e-2
beta1 = 0.9
beta2 = 0.999
weight_decay = 5e-4
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = 'cosine'
warmup_iters = 777  # warmup 3 epochs
warmup_bias_lr = 0.1
warmup_momentum = 0.8
lr_decay_iters = 77700  # should be ~= max_iters
min_lr = 1e-4  # minimum learning rate, should be ~= learning_rate/10
use_fused = True  # somehow use_fused=True is incompatible to compile=True in this model

# Eval related
# voc val set has 4,952 imgs, so 1 epoch ~= 78 iters
eval_interval = 777  # keep frequent if we'll overfit
eval_iters = 78  # use entire val set to get good estimate
score_thresh = 0.001
iou_thresh = 0.5  # for best map50

# Log related
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
out_dir = f'out/yolov3_voc/{timestamp}'
wandb_log = True
wandb_project = 'voc'
wandb_run_name = f'yolov3_{timestamp}'
log_interval = 10  # don't print too often
always_save_checkpoint = True  # only save when val improves if we expect overfit

# System related
compile = False  # somehow use_fused=True is incompatible to compile=True in this model
n_worker = 8
