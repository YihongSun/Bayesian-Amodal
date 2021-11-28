from configs import device_ids, dataset_train, dataset_eval, nn_type, vc_num, K, vMF_kappa, context_cluster, layer, exp_dir, categories, feature_num
from configs import *
from model import get_compnet_head
from DataLoader import Occ_Veh_Dataset, KINS_Dataset
from scipy import interpolate
from torch.utils.data import DataLoader
from util import roc_curve, rank_perf, visualize, draw_box, visualize_multi, calc_iou, print_, graph_prior
import copy
import warnings
import random
import datetime
import time
import sys
import torch
import torch.nn as nn
from losses import ClusterLoss, WeaklySupMaskLoss
warnings.filterwarnings("ignore", category=RuntimeWarning)


def create_artificial_inmodal_boxes(amodal_boxes, img_size, h_v_std=0.05, w_v_std=0.075, occ_std=0.7, max_occ=0.7):
    img_h, img_w = img_size
    for i in range(amodal_boxes.shape[0]):
        org_box = copy.deepcopy(amodal_boxes[i])
        h, w = amodal_boxes[i][2] - amodal_boxes[i][0], amodal_boxes[i][3] - amodal_boxes[i][1]

        varying_height = h * np.random.normal(0, h_v_std, 2)
        varying_width = w * np.random.normal(0, w_v_std, 2)

        amodal_boxes[i][0] = max(0, amodal_boxes[i][0] + varying_height[0])
        amodal_boxes[i][1] = max(0, amodal_boxes[i][1] + varying_width[0])
        amodal_boxes[i][2] = min(img_h, amodal_boxes[i][2] + varying_height[1])
        amodal_boxes[i][3] = min(img_w, amodal_boxes[i][3] + varying_width[1])

        h, w = amodal_boxes[i][2] - amodal_boxes[i][0], amodal_boxes[i][3] - amodal_boxes[i][1]

        artificial_occ = min(abs(np.random.normal(0, occ_std)), max_occ)
        vis_start = w * np.random.uniform(0, artificial_occ)
        vis_size = w * (1 - artificial_occ)
        amodal_boxes[i][1] = amodal_boxes[i][1] + vis_start
        amodal_boxes[i][3] = amodal_boxes[i][1] + vis_size

        if amodal_boxes[i][2] - amodal_boxes[i][0] <= 0 or amodal_boxes[i][3] - amodal_boxes[i][1] <= 0:
            amodal_boxes[i] = org_box

        if (amodal_boxes[i][2] - amodal_boxes[i][0]) / (org_box[2] - org_box[0]) < 0.5 or (amodal_boxes[i][2] - amodal_boxes[i][0]) / (org_box[2] - org_box[0]) > 2:
            amodal_boxes[i] = org_box

def visualize_model_prior(model, dir, name):
    prior_dir = dir + name + '/'
    if not os.path.exists(prior_dir):
        os.mkdir(prior_dir)

    for i, category in enumerate(categories['train']):
        fg_prior_set = model.fg_prior[i]
        context_prior_set = model.context_prior[i]

        for k in range(fg_prior_set.shape[0]):
            graph_prior(fg_prior_set[k].detach().cpu().numpy(), context_prior_set[k].detach().cpu().numpy(), name='{}{}_{}'.format(prior_dir, category, k))


def eval(model, test_loader, header=''):
    model.train = False
    correct = 0
    ious = []
    N = len(test_loader.dataset)
    for index, data in enumerate(test_loader):

        input_tensor, gt_labels, gt_inmodal_bbox, gt_amodal_bbox, gt_inmodal_segmentation, gt_amodal_segmentation, gt_occ, demo_img, img_path, failed = data
        if failed.item():
            continue

        input = input_tensor.cuda(device_ids[0])
        input.requires_grad = False
        label = gt_labels[0].numpy()
        gt_amodal_segmentation = gt_amodal_segmentation[0].numpy()

        if input_bbox_type == 'inmodal':
            input_bbox = copy.deepcopy(gt_inmodal_bbox[0])
        elif input_bbox_type == 'amodal':
            input_bbox = copy.deepcopy(gt_amodal_bbox[0])
        else:
            input_bbox = None
            print('input_bbox_type not recognized')

        create_artificial_inmodal_boxes(input_bbox, input_tensor.shape[2:])
        input_alighnments = []
        for bi, box in enumerate(input_bbox):
            box = box.type(torch.FloatTensor)
            gt_amodal_bbox[0] = gt_amodal_bbox[0].type(torch.FloatTensor)
            input_alighnments.append(((np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]) - np.array(
                [(gt_amodal_bbox[0][bi][0] + gt_amodal_bbox[0][bi][2]) / 2,
                 (gt_amodal_bbox[0][bi][1] + gt_amodal_bbox[0][bi][3]) / 2])) / (box[2] - box[0]) * 14).type(
                torch.IntTensor))

        with torch.no_grad():
            pred_scores, pred_confidence, pred_amodal_bboxes, pred_segmentations = model(org_x=input, bboxes=input_bbox, bbox_type=input_bbox_type, input_label=None, mask_label_training=gt_labels[0].numpy(), gt_alignment=input_alighnments)

        for obj_idx in range(pred_scores.shape[0]):
            label_correct = int(pred_scores[obj_idx].argmax() == label[obj_idx])
            correct += label_correct

            gt_seg = gt_amodal_segmentation[obj_idx]
            pred_seg = (pred_segmentations[obj_idx]['amodal'] >= 0.01).astype(float)

            ious.append(np.sum(pred_seg * gt_seg) / np.sum((pred_seg + gt_seg > 0.01).astype(int)))
        if index % 10 == 0:
            print('{}: {}/{}'.format(header, index, N), end='\r')
            visualize_multi(demo_img[0].numpy(), [(pred_segmentations[0]['amodal'] >= 0.01).astype(float), gt_amodal_segmentation[0]],
                            'temp/{}_{}'.format(header, index), cbar=True)


    acc = correct / len(ious)
    meanIoU = np.mean(ious)

    print_('{}:   Eval Acc: {:.3f}   Eval meanIoU: {:.3f}'.format(header, acc, meanIoU), file=file)
    model.train = True
    return acc, meanIoU



def train(model, train_loader, test_loader, epochs, batch_size, learning_rate, savedir, alpha=3, beta=3, gamma=1, vc_flag=False, mix_flag=False, prior_flag=False):
    best_check = {
        'cur_epoch': 0,
        'cur_acc': 0,
        'cur_mIoU': 0,
        'best_epoch': 0,
        'best_acc': 0,
        'best_mIoU': 0,
    }

    # we observed that training the backbone does not make a very big difference but not training saves a lot of memory
    # if the backbone should be trained, then only with very small learning rate e.g. 1e-7
    for param in model.extractor.parameters():
        param.requires_grad = False

    model.clutter_conv1o1.weight.requires_grad = False
    model.exp.vMF_kappa.requires_grad = False

    model.vc_conv1o1.weight.requires_grad = vc_flag

    for i in range(len(model.fg_models)):
        model.fg_models[i].requires_grad = mix_flag
        model.context_models[i].requires_grad = mix_flag
        model.fg_prior[i].requires_grad = prior_flag
        model.context_prior[i].requires_grad = prior_flag

    classification_loss = nn.CrossEntropyLoss()
    cluster_loss = ClusterLoss()
    mask_loss = WeaklySupMaskLoss(cls_loss=classification_loss, mil_coef=1, pairwise_coef=0.05, sampling_fraction=1)

    optimizer = torch.optim.Adagrad(params=filter(lambda param: param.requires_grad, model.parameters()),
                                    lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)

    print('Training...')
    if prior_flag:
        prior_dir = overall_trn_dir + 'prior/'
        if not os.path.exists(prior_dir):
            os.mkdir(prior_dir)

    for epoch in range(epochs):
        if epoch == 0:
            if prior_flag:
                visualize_model_prior(model, prior_dir, 'init')
            acc, meanIoU = eval(model, test_loader, header='Initial Eval')
            best_check['best_acc'] = acc
            best_check['best_mIoU'] = meanIoU

        start = time.time()
        model.train = True
        model.extractor.eval()
        N = len(train_loader.dataset)

        train_loss, train_loss_ = 0.0, 0.0
        correct, correct_ = 0, 0
        total, total_ = 0, 0

        for index, data in enumerate(train_loader):
            if index % 500 == 0 and index != 0:
                train_loss += train_loss_
                correct += correct_
                total += total_
                end = time.time()
                print('Epoch{:2}: {:4}/{:4},  Acc: {:.2f} ({:.2f}),  Loss: {:.2f} ({:.2f}),  Time:{:.2f}'.format(epoch + 1, index, N, correct_.cpu().item() / total_, correct.cpu().item() / total, train_loss_.cpu().item() / total_, train_loss.cpu().item() / total, (end - start)))
                start = time.time()

                train_loss_ = 0.0
                correct_ = 0
                total_ = 0

            input_tensor, gt_labels, gt_inmodal_bbox, gt_amodal_bbox, gt_inmodal_segmentation, gt_amodal_segmentation, gt_occ, demo_img, img_path, failed = data
            if failed.item():
                continue

            input = input_tensor.cuda(device_ids[0])
            label = gt_labels.cuda(device_ids[0])

            if input_bbox_type == 'inmodal':
                input_bbox = copy.deepcopy(gt_inmodal_bbox[0])
            elif input_bbox_type == 'amodal':
                input_bbox = copy.deepcopy(gt_amodal_bbox[0])
            else:
                input_bbox = None
                print('input_bbox_type not recognized')

            create_artificial_inmodal_boxes(input_bbox, input_tensor.shape[2:])
            input_alighnments = []
            for bi, box in enumerate(input_bbox):
                box = box.type(torch.FloatTensor)
                gt_amodal_bbox[0] = gt_amodal_bbox[0].type(torch.FloatTensor)
                input_alighnments.append(((np.array([(box[0] + box[2])/2, (box[1] + box[3])/2]) - np.array([(gt_amodal_bbox[0][bi][0] + gt_amodal_bbox[0][bi][2])/2, (gt_amodal_bbox[0][bi][1] + gt_amodal_bbox[0][bi][3])/2])) / (box[2] - box[0]) * 14).type(torch.IntTensor))

            pred_likelihood, pred_scores, pred_amodal_bboxes, pred_segmentations = net(org_x=input, bboxes=input_bbox, bbox_type=input_bbox_type, mask_label_training=gt_labels[0].numpy(), gt_alignment=input_alighnments)
            loss = classification_loss(pred_scores, label[0])

            for i, box in enumerate(input_bbox):
                if vc_flag:
                    vgg_feat = net.get_feature_activation(input[:, :, box[0]:box[2], box[1]:box[3]], resize=True)
                    loss += alpha * cluster_loss(vgg_feat, model.vc_conv1o1.weight)

                if mix_flag:
                    l = -1 * beta * pred_likelihood[i][label[0][i]]
                    loss += l.item()

                loss += gamma * mask_loss(pred_segmentations[i]['amodal_raw'], gt_amodal_bbox[0][i])

            loss = loss / input_bbox.shape[0]

            correct_ += torch.sum(pred_scores.argmax(1) == label)
            total_ += input_bbox.shape[0]

            # with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            # pseudo batches

            if np.mod(index, batch_size) == 0:  # and index!=0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss_ += loss.detach() * input.shape[0]

        scheduler.step()
        train_acc = correct.cpu().item() / total
        train_loss = train_loss.cpu().item() / total
        out_str = 'Epochs: [{:4}/{:4}],  Train Acc: {:.3f},  Train Loss: {:.3f}'.format(epoch + 1, epochs, train_acc, train_loss)
        print_(out_str, file=file)

        net.update_fused_models(omega=omega)
        acc, meanIoU = eval(model, test_loader, header='Eval')
        best_check['cur_epoch'] = epoch + 1
        best_check['cur_acc'] = acc
        best_check['cur_mIoU'] = meanIoU
        if prior_flag:
            visualize_model_prior(model, prior_dir, 'epoch_{}'.format(epoch + 1))

        if best_check['cur_acc'] > 0.95 * best_check['best_acc'] or best_check['cur_mIoU'] > 0.95 * best_check['best_mIoU']:
            best_check['best_acc'] = max(best_check['best_acc'], best_check['cur_acc'])
            best_check['best_mIoU'] = max(best_check['cur_mIoU'], best_check['best_mIoU'])

            now = datetime.datetime.now()
            save_path = '{}{}_{}_{}_{}_{:.4f}_{:.4f}.pth'.format(savedir, now.month, now.day, now.hour, now.minute, best_check['cur_acc'], best_check['cur_mIoU'])
            print_('BEST:   Saving path to {}'.format(save_path), file=file)
            save_check = {
                'state_dict': model.state_dict(),
                'val_acc': best_check['cur_acc'],
                'val_mIoU': best_check['cur_mIoU'],
                'epoch': epoch + 1
            }
            torch.save(save_check, save_path)



        print_('\n\n', file=file)


    return best_check



if __name__ == '__main__':

    tag = '11_7_align_1g'
    input_bbox_type = 'amodal'     # inmodal,  amodal

    alpha = 0   # vc-loss
    beta = 2    # mix loss
    gamma = 1   # mask loss
    lr = 5e-3  # learning rate
    batch_size = 1  # these are pseudo batches as the aspect ratio of images for CompNets is not square

    # Training setup
    vc_flag = False  # train the vMF kernels
    mix_flag = True  # train mixture components
    prior_flag = False  # train mixture components
    epochs = 15  # number of epochs to train


    dataType = 'train'
    omega = 0.2

    overall_trn_dir = trn_dir + '{}_{}/'.format(tag, dataset_train)
    if not os.path.exists(overall_trn_dir):
        os.mkdir(overall_trn_dir)

    checkpoint_dir = overall_trn_dir + 'checkpoints/'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    file = open(overall_trn_dir + 'training_info.txt', 'w')

    if dataset_eval == 'occveh':
        net = get_compnet_head(mix_model_dim_reduction=True, mix_model_suffix='_RP_newer_it2')
    elif dataset_eval == 'kinsv':
        net = get_compnet_head(mix_model_dim_reduction=True, mix_model_suffix='_cross_domain', dataset_override='pascal3d+')

    print_('\n\n', file=file)
    print_('Training Tag: {}'.format(tag), file=file)

    if len(sys.argv) > 1:
        pretrain_file = sys.argv[1]
        net.load_state_dict(torch.load(pretrain_file, map_location='cuda:{}'.format(device_ids[0]))['state_dict'])
        print_('Loaded Pretrain Model: {}'.format(pretrain_file))
    net.update_fused_models(omega=omega)


    if dataset_eval == 'occveh':
        train_set = Occ_Veh_Dataset(cats=categories['occveh'], dataType='train', train_types=['raw'], fg_level=0,
                                   bg_level=0, single_obj=True, resize=False, crop_img=False, crop_padding=48,
                                   crop_central=False, data_range=[0,0.9], demo_img_return=True)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

        test_set = Occ_Veh_Dataset(cats=categories['occveh'], dataType='train', train_types=['raw'], fg_level=0,
                                    bg_level=0, single_obj=True, resize=False, crop_img=False, crop_padding=48,
                                    crop_central=False, data_range=[0.9, 1], demo_img_return=True)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    elif dataset_eval == 'kinsv':
        height_thrd = 50
        train_set = KINS_Dataset(category_list=categories['kinsv'], dataType='train', occ=[0,1], height_thrd=height_thrd,       #categories['train']
                                amodal_height=True, data_range=[0,0.9], demo_img_return=True)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

        test_set = KINS_Dataset(category_list=categories['kinsv'], dataType='train', occ=[0, 1],
                                 height_thrd=height_thrd,
                                 amodal_height=True, data_range=[0.9, 1], demo_img_return=True)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    else:
        train_loader, test_loader = None, None

    train(model=net, train_loader=train_loader, test_loader=test_loader, epochs=epochs, batch_size=batch_size, learning_rate=lr, savedir=checkpoint_dir, alpha=alpha, beta=beta, gamma=gamma, vc_flag=vc_flag, mix_flag=mix_flag, prior_flag=prior_flag)

    file.close()
