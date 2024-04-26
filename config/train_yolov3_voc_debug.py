# Config for training YOLOv3 model on Pascal VOC 2007&2012 VOC dataset for object detection
# Train on VOC 2007 trainval and 2012 trainval, and evaluate on nano VOC 2007 test
import time

# Task related
task_name = 'detect'
init_from = 'backbone'
from_ckpt = 'saved/darknet53-448_imagenet2012/20240404-112658/ckpt_last.pt'  # TODO: try 'saved/pjreddie/darknet53_448.pt'

# Data related
dataset_name = 'voc'
img_h = 416  # TODO: increase to 640
img_w = 416  # TODO: increase to 640
n_class = 20

# Transform related
aug_type = 'default'
letterbox = True
fill = (123.0, 117.0, 104.0)
color_p = 1.0
brightness = 0.6
contrast = 0.6
saturation = 0.7
hue = 0.3
blur_p = 0.1
blur_size_min = 3
blur_size_max = 7
blur_sigma_min = 0.1
blur_sigma_max = 2.0
autocontrast_p = 0.1
posterize_p = 0.0
posterize_bits = 4
grayscale_p = 0.1
channelshuffle_p = 0.0
perspective_p = 1.0
perspective = 0.0
translate = 0.3
scale = 0.75
shear_p = 1.0
shear = 5.0
rotate_p = 1.0
degrees = 5.0
crop_scale = 0.75
ratio_min = 0.5
ratio_max = 2.0
flip_p = 0.5
min_size = 1.0
imgs_mean = (0.0, 0.0, 0.0)
imgs_std = (1.0, 1.0, 1.0)
multiscale_h = (416,)  # TODO: enable multisclae
multiscale_w = (416,)  # TODO: enable multisclae

# Model related
model_name = 'yolov3'
n_scale = 3
n_anchor_per_scale = 3
anchors = (((10, 13), (16, 30), (33, 23)), ((30, 61), (62, 45), (59, 119)), ((116, 90), (156, 198), (373, 326)))
init_weight = True  # TODO: try False

# Loss related
match_by = 'wh_iou'  # TODO: try wh_iou
match_thresh = 0.2  # from ultralytics/yolov3: hyp.VOC.yaml
iou_loss_type = 'giou'
rescore = 1.0
smooth = 0.0
pos_weight_class = 0.5  # from ultralytics/yolov3: hyp.VOC.yaml
pos_weight_obj = 0.67198  # from ultralytics/yolov3: hyp.VOC.yaml
balance = (4.0, 1.0, 0.4)
lambda_box = 0.02  # from ultralytics/yolov3: hyp.VOC.yaml, no scale needed
lambda_obj = 0.51728 * ((416*416)/(640*640))  # from ultralytics/yolov3: hyp.VOC.yaml, scaled from area 640*640 to 416*416TODO
lambda_class = 0.21638 * (20/80)  # from ultralytics/yolov3: hyp.VOC.yaml, scaled from 80 classes to 20TODO classes

# Train related
# the number of examples per iter:
# 64TODO batch_size * 1TODO grad_accum = 64 imgs/iter
# voc train set has 16,551 imgs, so 1 epoch ~= 259TODO iters
gradient_accumulation_steps = 1
batch_size = 64  # TODO: filled up the gpu memory on my machine
max_iters = 77700  # 300 epochs, finish in TODO hr on my machine  # TODO: increase

# Optimizer related  # TODO: try all hyp.VOC.yaml except iters
optimizer_type = 'adam'
learning_rate = 1e-4
beta1 = 0.74832  # from ultralytics/yolov3: hyp.VOC.yaml
beta2 = 0.999
weight_decay = 5e-4
grad_clip = 10.0  # clip gradients at this value, or disable if == 0.0
decay_lr = 'cosine'  # TODO: try step
warmup_iters = 777  # warmup 3 epochs
warmup_bias_lr = 0.18657  # from ultralytics/yolov3: hyp.VOC.yaml
warmup_momentum = 0.59462  # from ultralytics/yolov3: hyp.VOC.yaml
lr_decay_iters = 77700  # should be ~= max_iters
min_lr = 1e-6  # minimum learning rate, should be ~= learning_rate/10  # TODO: try 0.0
use_fused = True  # somehow use_fused=True is incompatible to compile=True in this model

# Eval related
# voc val set has 4,952 imgs, so 1 epoch ~= 78TODO iters
eval_interval = 777  # keep frequent if we'll overfit  # TODO: <= warmup_iters, >= 1 epoch
eval_iters = 78  # use entire val set to get good estimate  # TODO: decrease to speedup
score_thresh = 0.001
iou_thresh = 0.5  # for best map50
max_n_pred_per_img = 1000  # TODO: try 10000

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
n_worker = 8  # TODO: tune
