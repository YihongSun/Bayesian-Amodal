from sklearn.cluster import SpectralClustering
from torch.utils.data import DataLoader
import gc
import torch
from configs import device_ids, dataset_train, dataset_eval, nn_type, vc_num, K, vMF_kappa, context_cluster, layer, meta_dir, categories, feature_num, feat_stride
from configs import *
from DataLoader import get_pascal3d_data, get_coco_data, Single_Object_Loader, KINS_Compnet_Train_Dataset
from util import roc_curve, rank_perf, visualize, res_down, graph_prior
from model import get_compnet_head, get_mixture_models
from old_Net_E2E import Conv1o1Layer
import pickle
from datetime import datetime

# datetime object containing current date and time


def get_cls_perf(net, data_loader):
    running_mean = 0
    for ii, data in enumerate(data_loader):
        if dataset_train == 'pascal3d+':
            org_x, label, bboxes, gt_mask, demo_img, img_path, true_pad = data
            true_pad = true_pad[0].item()

        bboxes[0, 0] = true_pad
        bboxes[0, 1] = true_pad
        bboxes[0, 2] = org_x.shape[2] - true_pad
        bboxes[0, 3] = org_x.shape[3] - true_pad
        with torch.no_grad():
            score, center, mixture, amodal_bboxes, bundle = net.classify_init(org_x.cuda(device_ids[0]), bboxes, slide_window=False, gt_labels=None, pad_length=48)
        pred_cat = torch.argmax(score[0]).item()
        gt_cat = label[0].item()

        running_mean = running_mean * (ii/(ii+1)) + int(pred_cat==gt_cat) * (1/(ii+1))
        if ii % 10 == 9:
            print('{} - Running Mean: {:.2f}%'.format(ii+1, running_mean*100), end='\r')
        
    print('Mean:', running_mean, '                            ')
    print()

net = get_compnet_head(mix_model_dim_reduction=True, mix_model_suffix='_RP_newer_it2')
net.update_fused_models(type='standard')

if dataset_train == 'pascal3d+':
    image_files, mask_files, labels, bboxs = get_pascal3d_data(cats=categories['train'], train=True, single_obj=True)
    train_data_set = Single_Object_Loader(image_files, mask_files, labels, bboxs, resize=True, crop_img=True, crop_padding=48, crop_central=True, demo_img_return=True, return_true_pad=True)

    image_files, mask_files, labels, bboxs = get_pascal3d_data(cats=categories['train'], train=False, single_obj=True)
    test_data_set = Single_Object_Loader(image_files, mask_files, labels, bboxs, resize=True, crop_img=True, crop_padding=48, crop_central=True, demo_img_return=True, return_true_pad=True)

if dataset_train == 'kinsv':
    assert False, 'Not implemented'
    data_set = KINS_Compnet_Train_Dataset(category=category, height_thrd=75, pad=48)

train_size = train_data_set.__len__()
print('Pre-learning Train Cls Perf -' , train_size)
train_data_loader = DataLoader(dataset=train_data_set, batch_size=1, shuffle=True)
# get_cls_perf(net, train_data_loader)

print('Pre-learning Test Cls Perf -' , test_data_set.__len__())
test_data_loader = DataLoader(dataset=test_data_set, batch_size=1, shuffle=True)
# get_cls_perf(net, test_data_loader)

train_perc = 1.0

old_vcs = net.vc_conv1o1.weight.data
new_vcs = torch.zeros_like(old_vcs) #(512, 1024, 1, 1)
alpha_sum = torch.zeros(new_vcs.shape[0]).cuda(device_ids[0])

for ii, data in enumerate(train_data_loader):
    with torch.no_grad():

        ## ================ TEMP ================ ##
        if ii % 100 == 1:
            print(ii)
            score, center, mixture, amodal_bboxes, bundle = net.classify_init(org_x.cuda(device_ids[0]), bboxes, slide_window=False, pad_length=48)
            print(score, gt_cat)
            new_vcs_ = new_vcs / (alpha_sum.reshape(-1, 1, 1, 1)+1e-8)
            new_vcs_ = new_vcs_ / (torch.sqrt(torch.sum(new_vcs_ ** 2, dim=1, keepdim=True))+1e-8)
            net.vc_conv1o1 = Conv1o1Layer(new_vcs_)

            score, center, mixture, amodal_bboxes, bundle = net.classify_init(org_x.cuda(device_ids[0]), bboxes, slide_window=False, pad_length=48)
            print(score, torch.argmax(score[0]).item())

            net.vc_conv1o1 = Conv1o1Layer(old_vcs)

        ## ================ TEMP ================ ##

        if dataset_train == 'pascal3d+':
            org_x, label, bboxes, gt_mask, demo_img, img_path, true_pad = data
            true_pad = true_pad[0].item()

        bboxes[0, 0] = true_pad
        bboxes[0, 1] = true_pad
        bboxes[0, 2] = org_x.shape[2] - true_pad
        bboxes[0, 3] = org_x.shape[3] - true_pad
        gt_cat = label[0].item()
        
        score, center, mixture, amodal_bboxes, bundle = net.classify_init(org_x.cuda(device_ids[0]), bboxes, slide_window=False, gt_labels=[gt_cat], pad_length=48)

        k_max = mixture[0, gt_cat]
        feat, vc_conv, vc_act = bundle

        feat = feat[0]
        alpha = net.fused_models[gt_cat][k_max]

        min_h, min_w = min(feat.shape[1], alpha.shape[1]), min(feat.shape[2], alpha.shape[2])

        feat = net.center_crop(feat, [min_h, min_w])
        alpha = net.center_crop(alpha, [min_h, min_w])

        feat = feat.permute(1, 2, 0).reshape(-1, new_vcs.shape[1])      #(B, 1024)
        alpha = alpha.permute(1, 2, 0).reshape(-1, new_vcs.shape[0])    #(B, 512)

        # weighted sum solution
        # tmp_vc = feat.unsqueeze(1) * alpha.unsqueeze(2)
        # new_vcs = new_vcs + tmp_vc.sum(0).reshape(*new_vcs.shape)
        # alpha_sum = alpha_sum + alpha.sum(0)

        # one hot solution
        alpha_max = torch.argmax(alpha, 1)
        for i in range(alpha_max.shape[0]):
            new_vcs[alpha_max[i]] = new_vcs[alpha_max[i]] + feat[i].unsqueeze(1).unsqueeze(2)
            alpha_sum[alpha_max[i]] = alpha_sum[alpha_max[i]] + 1



        if ii % 10 == 9:
            print(ii, end='\r')
        
        if ii > train_size * train_perc:
            break

new_vcs = new_vcs / (alpha_sum.reshape(-1, 1, 1, 1)+1e-8)
new_vcs = new_vcs / (torch.sqrt(torch.sum(new_vcs ** 2, dim=1, keepdim=True))+1e-8)

dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M")
torch.save(new_vcs, 'new_vcs_{}.pt'.format(dt_string))
print('New VCs saved as \' {} \''.format('new_vcs_{}.pt'.format(dt_string)))

net.vc_conv1o1 = Conv1o1Layer(new_vcs)

print('Post-learning Train Cls Perf -' , train_size)
get_cls_perf(net, train_data_loader)

print('Post-learning Test Cls Perf -' , test_data_set.__len__())
get_cls_perf(net, test_data_loader)

