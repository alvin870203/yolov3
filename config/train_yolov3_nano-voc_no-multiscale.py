# Config for training YOLOv3 model on single-scale nano VOC dataset for object detection debug
# Train on nano VOC 2007 trainval and 2012 trainval, and evaluate on nano VOC 2007 test
import time

# Task related
task_name = 'detect'
init_from = 'backbone'
from_ckpt = 'saved/darknet53-448_imagenet2012/20240404-112658/ckpt_last.pt'

# Data related
dataset_name = 'nano_voc'
img_h = 416
img_w = 416
n_class = 20

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
scale = 0.75
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
multiscale_h = (416,)
multiscale_w = (416,)

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
lambda_obj = 1.0
lambda_class = 0.5

# Train related
# the number of examples per iter:
# 2 batch_size * 1 grad_accum = 2 imgs/iter
# nano voc train set has 5 imgs, so 1 epoch ~= 3 iters
gradient_accumulation_steps = 1
batch_size = 2
max_iters = 3000

# Optimizer related
optimizer_type = 'adamw'
learning_rate = 1e-4  # nano lr to overfit nano dataset stably
beta1 = 0.9
beta2 = 0.999
weight_decay = 1e-4
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = False  # no lr scheduling
#warmup_iters = 5  # warmup 5 epochs
#warmup_bias_lr = 0.1
#warmup_momentum = 0.8
#lr_decay_iters = 3000  # should be ~= max_iters
#min_lr = 0  # minimum learning rate, should be ~= learning_rate/10
use_fused = True  # somehow use_fused=True is incompatible to compile=True in this model

# Eval related
# voc val set has 5 imgs, so 1 epoch ~= 3 iters
eval_interval = 3  # keep frequent if we'll overfit
eval_iters = 3  # use entire val set to get good estimate
score_thresh = 0.001
iou_thresh = 0.5  # for best map50
max_n_pred_per_img = 1000

# Log related
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
out_dir = f'out/yolov3_nano-voc/{timestamp}'
wandb_log = True
wandb_project = 'nano-voc'
wandb_run_name = f'yolov3_{timestamp}'
log_interval = 1  # don't print too often
always_save_checkpoint = False  # only save when val improves if we expect overfit

# System related
compile = False  # somehow use_fused=True is incompatible to compile=True in this model
n_worker = 4
