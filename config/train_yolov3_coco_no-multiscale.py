# Config for training YOLOv3 model on COCO dataset for object detection
# Train on COCO 2017 train, and evaluate on COCO 2017 val
import time

# Task related
task_name = 'detect'
init_from = 'backbone'
from_ckpt = 'saved/pjreddie/darknet53_448.pt'

# Data related
dataset_name = 'coco'
img_h = 640
img_w = 640
n_class = 80

# Transform related
aug_type = 'default'
letterbox = True
fill = (123.0, 117.0, 104.0)
color_p = 1.0
brightness = 0.4
contrast = 0.4
saturation = 0.7
hue = 0.015
blur_p = 0.01
blur_size_min = 3
blur_size_max = 7
blur_sigma_min = 0.1
blur_sigma_max = 2.0
autocontrast_p = 0.01
posterize_p = 0.0
posterize_bits = 4
grayscale_p = 0.01
channelshuffle_p = 0.0
perspective_p = 1.0
perspective = 0.0
translate = 0.1
scale = 0.5
shear_p = 1.0
shear = 0.0
rotate_p = 1.0
degrees = 0.0
crop_scale = 0.8
ratio_min = 0.5
ratio_max = 2.0
flip_p = 0.5
min_size = 1.0
imgs_mean = (0.0, 0.0, 0.0)
imgs_std = (1.0, 1.0, 1.0)
multiscale_h = (640,)
multiscale_w = (640,)

# Model related
model_name = 'yolov3'
n_scale = 3
n_anchor_per_scale = 3
anchors = (((10, 13), (16, 30), (33, 23)), ((30, 61), (62, 45), (59, 119)), ((116, 90), (156, 198), (373, 326)))
init_weight = True
init_bias = True

# Loss related
match_by = 'wh_ratio'
match_thresh = 4.0
iou_loss_type = 'ciou'
rescore = 1.0
smooth = 0.0
pos_weight_class = 1.0
pos_weight_obj = 1.0
balance = (4.0, 1.0, 0.4)
lambda_box = 0.05
lambda_obj = 1.0 * ((640*640)/(640*640))  # scaled from area 640*640 to 640*640
lambda_class = 0.5 * (80/80)  # scaled from 80 classes to 80 classes

# Train related
# the number of examples per iter:
# 32 batch_size * 1 grad_accum = 32 imgs/iter
# coco train set has 118,287 imgs, so 1 epoch ~= 3,697 iters
gradient_accumulation_steps = 1
batch_size = 32  # filled up the gpu memory on my machine
max_iters = 1109100  # 300 epochs, finish in 112 hr on my machine

# Optimizer related  # from ultralytics/yolov3: hyp.scratch-low.yaml
optimizer_type = 'sgd'
learning_rate = 1e-2
beta1 = 0.937
beta2 = 0.999
weight_decay = 5e-4
grad_clip = 10.0  # clip gradients at this value, or disable if == 0.0
decay_lr = 'cosine'
warmup_iters = 11091  # warmup 3 epochs
warmup_bias_lr = 0.1
warmup_momentum = 0.8
lr_decay_iters = 1109100  # should be ~= max_iters
min_lr = 1e-4  # minimum learning rate, should be ~= learning_rate/10
use_fused = True  # somehow use_fused=True is incompatible to compile=True in this model

# Eval related
# coco val set has 5,000 imgs, so 1 epoch ~= 157 iters
eval_interval = 11091  # keep frequent if we'll overfit
eval_iters = 157  # use entire val set to get good estimate
score_thresh = 0.001
iou_thresh = 0.5  # for best map50
max_n_pred_per_img = 1000

# Log related
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
out_dir = f'out/yolov3_coco/{timestamp}'
wandb_log = True
wandb_project = 'coco'
wandb_run_name = f'yolov3_{timestamp}'
log_interval = 10  # don't print too often
always_save_checkpoint = True  # only save when val improves if we expect overfit

# System related
compile = False  # somehow use_fused=True is incompatible to compile=True in this model
n_worker = 8
