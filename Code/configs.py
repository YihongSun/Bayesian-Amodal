import os, cv2, pickle
import numpy as np
import matplotlib.pyplot as plt


### ==== Setups ==== ###

device_ids = [0]

TABLE_NUM = 1           #   1, 2, 3
MODEL_TYPE = 'ML'      #   ML, E2E      



if TABLE_NUM == 1 or TABLE_NUM == 3:
    EXPERIMENT_DATASET = 'occveh'
elif TABLE_NUM == 2:
    EXPERIMENT_DATASET = 'kinsv'
else:
    raise Exception('TABLE_NUM {} not recognized'.format(TABLE_NUM))



dataset_train = EXPERIMENT_DATASET
dataset_eval = EXPERIMENT_DATASET

if EXPERIMENT_DATASET == 'occveh':
    dataset_train = 'pascal3d+'

nn_type = 'resnext'             # vgg, resnext

vc_num = 512
K = 8
context_cluster = 5


### ==== Directories ==== ###

home_dir = '../'
meta_dir = home_dir + 'Models/'
data_dir = home_dir + 'Dataset/'
init_dir = meta_dir + 'ML_{}/'.format(nn_type)
exp_dir = home_dir + 'log/'
trn_dir = home_dir + 'training/'

for d in [exp_dir, trn_dir]:
    if not os.path.exists(d):
        os.mkdir(d)

### ==== Categories ==== ###

categories = dict()
categories['pascal3d+'] = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
categories['occveh'] = ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike']
categories['occobj'] = ['boat', 'bottle', 'chair', 'diningtable', 'sofa', 'tvmonitor']
categories['coco'] = ["None", "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", " parking meter", "bench", "bird", "cat", "dog",
                      "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                      "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                      "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                      "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                      "teddy bear", "hair drier", "toothbrush"]
categories['kins'] = ['_', 'cyclist', 'pedestrian', '_', 'car', 'tram', 'truck', 'van', 'misc']
categories['kinsv'] = ['cyclist', 'car', 'tram', 'truck', 'van']

# categories['occveh'] = ['boat']

categories['train'] = categories[dataset_train]
categories['eval']  = categories[dataset_eval]


### ==== Network ==== ###

vMF_kappas = {'vgg_pool4_pascal3d+' : 30, 'resnext_second_pascal3d+' : 65, 'vgg_pool4_kinsv' : 30, 'resnext_second_kinsv' : 50 }

if nn_type == 'vgg':
    layer = 'pool4'
    feature_num = 512
    feat_stride = 16

elif nn_type == 'resnext':
    layer = 'second'
    feature_num = 1024
    feat_stride = 16

else:
    print('Backbone Architecture Not Recognized')
    layer = ''
    feature_num = 0
    feat_stride = 0

vMF_kappa = vMF_kappas['{}_{}_{}'.format(nn_type, layer, dataset_train)]

rpn_configs = {'training_param' : {'weight_decay': 0.0005, 'lr_decay': 0.1, 'lr': 1e-3}, 'ratios' : [0.5, 1, 2], 'anchor_scales' : [8, 16, 32], 'feat_stride' : feat_stride }



'''

device_ids:         cuda device ids used
dataset_train:      dataset used for training
dataset_eval:       dataset used for evaluation
nn_type:            architecture type of backbone extractor
layer:              architecture layer of backbone extractor
feature_num:        number of channels of the architecture layer of backbone extractor
vMF_kappa:          the value used for the vMF activation
vc_num:             number of VC centers
context_cluster:    number of context centers
K:                  number of mixtures per category learned
categories:         map that contains various categories depending on the dataset
rpn_configs:        configs for the rpn training and evaluation
'''
