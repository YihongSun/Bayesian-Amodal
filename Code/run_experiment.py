from configs import device_ids, dataset_train, dataset_eval, nn_type, vc_num, K, vMF_kappa, context_cluster, layer, exp_dir, categories, feature_num, TABLE_NUM, MODEL_TYPE
from configs import *
from model import get_compnet_head
from DataLoader import Occ_Veh_Dataset, KINS_Dataset, COCOA_Dataset
from scipy import interpolate
from torch.utils.data import DataLoader
from util import roc_curve, rank_perf, visualize, draw_box, visualize_multi, visualize_mask, calc_iou, find_max_iou
import copy
import sys
import torch
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def make_three_dimensional_demo(pixel_cls, pixel_cls_score):

    occ_label = 0
    fg_label = 1
    bg_label = 2

    try:
        occ_range = [np.min(pixel_cls_score[pixel_cls == occ_label]), np.max(pixel_cls_score[pixel_cls == occ_label])]
        if occ_range[1] - occ_range[0] == 0:
            occ_range = [0, 1]
    except:
        occ_range = [0, 1]

    try:
        fg_range = [np.min(pixel_cls_score[pixel_cls == fg_label]), np.max(pixel_cls_score[pixel_cls == fg_label])]
        if fg_range[1] - fg_range[0] == 0:
            fg_range = [0, 1]
    except:
        fg_range = [0, 1]

    try:
        bg_range = [np.min(pixel_cls_score[pixel_cls == bg_label]), np.max(pixel_cls_score[pixel_cls == bg_label])]
        if bg_range[1] - bg_range[0] == 0:
            bg_range = [0, 1]
    except:
        bg_range = [0, 1]

    # all_min = min(occ_range[0], fg_range[0], bg_range[0] )
    # all_max = max(occ_range[1], fg_range[1], bg_range[1] )
    # occ_range = [all_min, all_max]
    # fg_range = [all_min, all_max]
    # bg_range = [all_min, all_max]

    # treat an rbg image as three layers heatmap

    occ_layer = ( (pixel_cls == occ_label).astype(float) * (pixel_cls_score - occ_range[0]) / (occ_range[1] - occ_range[0]) * 255 ).astype(int)[:, :, np.newaxis]
    fg_layer  = ( (pixel_cls == fg_label).astype(float) * (pixel_cls_score - fg_range[0]) / (fg_range[1] - fg_range[0]) * 255 ).astype(int)[:, :, np.newaxis]
    bg_layer  = ( (pixel_cls == bg_label).astype(float) * (pixel_cls_score - bg_range[0]) / (bg_range[1] - bg_range[0]) * 255 ).astype(int)[:, :, np.newaxis]

    # cv2.imwrite('temp_pred_check_o.jpg', occ_layer)
    # cv2.imwrite('temp_pred_check_f.jpg', fg_layer)
    # cv2.imwrite('temp_pred_check_c.jpg', bg_layer)

    img = np.concatenate((fg_layer, bg_layer, occ_layer), axis=2)

    return img

def print_(str, file=None):
    print(str)
    if file:
        print(str, file=file)

def make_demo(data_loader, rank, img_index, obj_index, exp_set_up, out_dir=''):
    mask_type, input_bbox_type, input_gt_label, bmask_thrd = exp_set_up

    out_dir_m = out_dir + '{}/'.format(mask_type)
    if not os.path.exists(out_dir_m):
        os.mkdir(out_dir_m)

    out_dir_p = out_dir + 'pixel_cls/'
    if not os.path.exists(out_dir_p):
        os.mkdir(out_dir_p)

    input_label = None
    input_bbox = None
    gt_seg = None

    input_tensor, gt_labels, gt_inmodal_bbox, gt_amodal_bbox, gt_inmodal_segmentation, gt_amodal_segmentation, gt_occ, demo_img, img_path, failed = data_loader.dataset.__getitem__(img_index)


    gt_inmodal_bbox, gt_amodal_bbox = torch.tensor(gt_inmodal_bbox), torch.tensor(gt_amodal_bbox)

    if input_bbox_type == 'inmodal':
        input_bbox = copy.deepcopy(gt_inmodal_bbox)
    elif input_bbox_type == 'amodal':
        input_bbox = copy.deepcopy(gt_amodal_bbox)

    if input_gt_label:
        input_label = gt_labels.copy()
    try:
        with torch.no_grad():
            c, h, w = input_tensor.shape
            input_tensor = torch.tensor(input_tensor).view(1, c, h, w)

            pred_scores, pred_confidence, pred_amodal_bboxes, pred_segmentations = net(org_x=input_tensor.cuda(device_ids[0]),
                                                                                       bboxes=input_bbox[obj_index:obj_index+1],
                                                                                       bbox_type=input_bbox_type,
                                                                                       input_label=input_label[obj_index:obj_index+1])
    except:
        print('error')

    pred_segmentation = pred_segmentations[0]
    pixel_cls_b = pred_segmentation['pixel_cls']
    pixel_cls_score = pred_segmentation['pixel_cls_score']

    # box = np.copy(gt_amodal_bbox[obj_index])                      #TODO
    box = np.copy(pred_amodal_bboxes[0])
    height_box = box[2] - box[0]
    width_box = box[3] - box[1]

    box[0] = max(0, box[0] - 0.1 * height_box)
    box[1] = max(0, box[1] - 0.1 * width_box)
    box[2] = min(input_tensor.shape[2], box[2] + 0.1 * height_box)
    box[3] = min(input_tensor.shape[3], box[3] + 0.1 * width_box)
    box = box.astype(int)

    copy_img = draw_box(demo_img.copy(), gt_inmodal_bbox[obj_index], color=(255, 0, 0), thick=2)
    copy_img = draw_box(copy_img, pred_amodal_bboxes[0], color=(0, 255, 0), thick=2)
    demo_img = copy_img[box[0]:box[2], box[1]:box[3], :]
    pixel_cls_img = make_three_dimensional_demo(pixel_cls=pixel_cls_b, pixel_cls_score=pixel_cls_score)[box[0]:box[2],
        box[1]:box[3], :]

    if mask_type == 'amodal':
        gt_seg = gt_amodal_segmentation[obj_index]

    elif mask_type == 'inmodal':
        gt_seg = gt_inmodal_segmentation[obj_index]

    elif mask_type == 'occ':
        gt_seg = (gt_amodal_segmentation[obj_index] - gt_inmodal_segmentation[obj_index] > 0).astype(float)

    pred_seg = (pred_segmentation[mask_type] >= bmask_thrd).astype(float)

    visualize_multi(demo_img, [pred_seg[box[0]:box[2], box[1]:box[3]], gt_seg[box[0]:box[2], box[1]:box[3]]], '{}{}_img{}_obj{}'.format(out_dir_m, rank, img_index, obj_index))
    cv2.imwrite('{}img{}_obj{}.jpg'.format(out_dir_p, img_index, obj_index), np.concatenate((demo_img, pixel_cls_img), axis=0))


def same_filename(name1, name2):
    img_dir1, img_name1 = name1.split('/')[-2:]
    img_dir2, img_name2 = name2.split('/')[-2:]

    return (img_dir1 == img_dir2) and (img_name1 == img_name2)

def eval_performance(data_loader, rpn_results, demo=False, category='car', eval_modes=('inmodal', 'amodal'), input_bbox_type='inmodal', input_gt_label=True, search_for_file=True):
    N = data_loader.__len__()

    cls_acc = []
    amodal_bbox_iou = []
    amodal_height = []
    occ = []
    disp_iou, disp_counter = 0, 0

    mask_dict = dict()
    for eval in eval_modes:
        mask_dict[eval] = {'img_index': [], 'obj_index': [], 'iou': [], 'bmask_thrd': binary_mask_thrds[eval]}

    input_bbox = None
    input_label = None
    gt_seg = None
    RPN_PREDICTION = []

    for ii, data in enumerate(data_loader):

        input_tensor, gt_labels, gt_inmodal_bbox, gt_amodal_bbox, gt_inmodal_segmentation, gt_amodal_segmentation, gt_occ, demo_img, img_path, failed = data
        if failed.item():
            continue

        gt_labels, gt_occ = gt_labels[0].numpy(), gt_occ[0].numpy()
        gt_inmodal_bbox, gt_amodal_bbox = gt_inmodal_bbox[0], gt_amodal_bbox[0]
        gt_inmodal_segmentation, gt_amodal_segmentation = gt_inmodal_segmentation[0].numpy(), gt_amodal_segmentation[0].numpy()

        if input_bbox_type == 'inmodal':
            input_bbox = copy.deepcopy(gt_inmodal_bbox)
        elif input_bbox_type == 'amodal':
            input_bbox = copy.deepcopy(gt_amodal_bbox)

        if input_gt_label:
            input_label = gt_labels.copy()

        bad_bbox = False
        for bi, box in enumerate(input_bbox):
            if box[2] - box[0] == 0 or box[3] - box[1] == 0:
                bad_bbox = True
                break
        if bad_bbox:
            continue

        if TABLE_NUM == 1 or TABLE_NUM == 2:
            load_ii = ii
            if search_for_file:
                for load_ii in range(len(rpn_results)):
                    if same_filename(rpn_results[load_ii]['file_name'], img_path[0]): 
                        break
            else:
                if not same_filename(rpn_results[load_ii]['file_name'], img_path[0]):
                    print('error, skip')
                    continue
            
            rpn_input_box = []
            for box in input_bbox:
                box, iou = find_max_iou(rpn_results[load_ii]['bbox'], box.cpu().numpy())
                RPN_PREDICTION.append(iou)
                rpn_input_box.append(box.astype(int))
        elif TABLE_NUM == 3:
            rpn_input_box = []
            for box in input_bbox:
                img_name_sep = str(img_path[0]).split('/')
                img_name = '/home/yihong/workspace/dataset/COCO/' + '/'.join(img_name_sep[-2:]) 
                box, iou = find_max_iou(rpn_results[img_name]['bbox'], box.cpu().numpy())
                RPN_PREDICTION.append(iou)
                rpn_input_box.append(box.astype(int))

        rpn_input_box = torch.tensor(np.stack(rpn_input_box))

        try:
            with torch.no_grad():
                pred_scores, pred_confidence, pred_amodal_bboxes, pred_segmentations = net(org_x=input_tensor.cuda(device_ids[0]), bboxes=rpn_input_box, bbox_type=input_bbox_type, input_label=input_label)
        except:
            print('error')
            continue

        pred_acc = (gt_labels == pred_scores.argmax(1).cpu().numpy()).astype(float)

        for obj_idx in range(len(pred_amodal_bboxes)):

            pred_segmentation = pred_segmentations[obj_idx]

            if np.sum(gt_inmodal_segmentation[obj_idx]) == 0:
                continue

            for mask_type in eval_modes:
                bmask_thrd = mask_dict[mask_type]['bmask_thrd']

                if mask_type == 'amodal':
                    gt_seg = gt_amodal_segmentation[obj_idx]

                elif mask_type == 'inmodal':
                    gt_seg = gt_inmodal_segmentation[obj_idx]

                elif mask_type == 'occ':
                    gt_seg = (gt_amodal_segmentation[obj_idx] - gt_inmodal_segmentation[obj_idx] > 0).astype(float)

                pred_seg = pred_segmentation[mask_type]
                pred_seg_b = (pred_seg >= bmask_thrd).astype(float)

                iou = np.sum(pred_seg_b * gt_seg) / np.sum((pred_seg_b + gt_seg > 0.5).astype(int))
                mask_dict[mask_type]['iou'].append(iou)
                mask_dict[mask_type]['img_index'].append(ii)
                mask_dict[mask_type]['obj_index'].append(obj_idx)

                if mask_type == 'amodal':
                    disp_iou = disp_iou * disp_counter / (disp_counter + 1) + iou / (disp_counter + 1)
                    disp_counter += 1

            cls_acc.append(pred_acc[obj_idx])
            amodal_bbox_iou.append(calc_iou(pred_box=pred_amodal_bboxes[obj_idx].numpy(), gt_box=gt_amodal_bbox[obj_idx].numpy()))
            amodal_height.append(gt_amodal_bbox[obj_idx][2] - gt_amodal_bbox[obj_idx][0])
            occ.append(gt_occ[obj_idx])

        if ii % 10 == 0:
            print('    {}  -  eval {}/{}     amodal_iou={:.2f}%         '.format(category, ii, N, disp_iou * 100), end='\r')

    cls_acc = np.array(cls_acc)
    amodal_bbox_iou = np.array(amodal_bbox_iou)
    amodal_height = np.array(amodal_height)
    occ = np.array(occ)

    cls_acc_val = np.mean(cls_acc)

    amodal_bbox_iou_val_correct_label = np.mean(amodal_bbox_iou[cls_acc == 1])
    amodal_bbox_iou_val = np.mean(amodal_bbox_iou * cls_acc)

    output = dict()

    print('RPN_iou:', np.mean(RPN_PREDICTION))

    output['cls'] = {'amodal_bbox_iou': amodal_bbox_iou, 'cls_acc': cls_acc, 'amodal_height': amodal_height, 'occ': occ,
                     'cls_acc_val' : cls_acc_val, 'amodal_bbox_iou_val' : amodal_bbox_iou_val, 'amodal_bbox_iou_val_correct_label' : amodal_bbox_iou_val_correct_label, 'num_objects': cls_acc.shape[0]}

    print_line = '{:15}  - {:5} -  cls_acc: {:6.4f}     |     amodal_box_mIoU: {:6.4f}     '.format(category, output['cls']['num_objects'], cls_acc_val, amodal_bbox_iou_val)

    for mask_type in eval_modes:
        m_dict = mask_dict[mask_type]
        iou_perf = np.array(m_dict['iou'])
        iou_perf_cls = iou_perf * cls_acc

        output[mask_type] = {'average iou': np.mean(iou_perf_cls), 'precision': np.mean((iou_perf_cls >= 0.5).astype(float)), 'num_objects': iou_perf.shape[0]}
        print_line += '|     {}_mIoU: {:6.4f}     '.format(mask_type, output[mask_type]['average iou'])


        if demo:
            rank_ind = np.argsort(iou_perf_cls)[::-1]

            out_dir = demo_dir + '{}/'.format(sub_tag)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            i = 0
            for ri in rank_ind:
                if i % 50 == 0 or i < 75:
                    make_demo(data_loader, rank=i, img_index=m_dict['img_index'][ri], obj_index=m_dict['obj_index'][ri], exp_set_up=[mask_type, input_bbox_type, input_gt_label, mask_dict[mask_type]['bmask_thrd']], out_dir=out_dir)
                    print('    {}  -  demo {}/{}     '.format(category, i, iou_perf.shape[0]), end='\r')
                i += 1

    print_(print_line, file=file)

    return output


if __name__ == '__main__':

    ### === Default Settings === ###

    bool_gt_label = True          
    bool_demo_seg = False
    tag = 'final'
    fraction_to_load = 1

    dataType = 'test'
    eval_modes = ['inmodal', 'occ', 'amodal']
    binary_mask_thrds = {'amodal': 0.5, 'inmodal': 0.5, 'occ': 0.5}
    omega = 0.2

    net = get_compnet_head(mix_model_dim_reduction=True, mix_model_suffix='_final')

    ### === Default Assertions === ###

    for eval in eval_modes:
        assert eval in ['inmodal', 'occ', 'amodal']

    ### === Eval Dataset Settings === ###

    if TABLE_NUM == 1:
        fg_levels = [0, 1, 2, 3]            
        bg_levels = [1, 2, 3]
    elif TABLE_NUM == 2:
        height_thrd = 50
        occ_bounds = [[0, 0], [0.0001, 0.3], [0.3001, 0.6], [0.6001, 0.9]]   #[0, 0], [0.0001, 0.3], [0.3001, 0.6], [0.6001, 0.9]
    elif TABLE_NUM == 3:
        fg_levels = [0, 1, 2, 3]             
        bg_levels = [0]




    ### === Directory Settings === ###

    overall_exp_dir = exp_dir + 'Table_{}_Model_{}_mIoU_{}/'.format(TABLE_NUM, MODEL_TYPE, tag)

    if not os.path.exists(overall_exp_dir):
        os.mkdir(overall_exp_dir)

    plot_dir = overall_exp_dir + 'plot_{}/'.format(dataset_eval)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    demo_dir = overall_exp_dir + 'demo_{}/'.format(dataset_eval)
    if not os.path.exists(demo_dir):
        os.mkdir(demo_dir)


    ### === File Storages === ###

    file = open(overall_exp_dir + 'exp_info_{}.txt'.format(dataset_eval), 'w')

    if MODEL_TYPE == 'E2E':
        if TABLE_NUM == 1 or TABLE_NUM == 3:
            pretrain_file = '../Models/E2E/pascal_final.pth'

        elif TABLE_NUM == 2:
            pretrain_file = '../Models/E2E/kinsv_final.pth'

        net.load_state_dict(torch.load(pretrain_file, map_location='cuda:{}'.format(device_ids[0]))['state_dict'])
        print_('Loaded Pretrain Model: {}'.format(pretrain_file), file=file)
    net.update_fused_models(omega=omega)

    print_('\n\n', file=file)
    print_('Experimental Tag: {}'.format(tag), file=file)
    print_('{:25}{}\n{:25}{}\n{:25}{}\n{:25}{}'.format('Table:', TABLE_NUM, 'Model type:', MODEL_TYPE, 
        'Segmentation:', eval_modes, 'Backbone:', nn_type,), file=file)
    net.remove_individual_models()

    try:
        with open(overall_exp_dir + '/exp_meta_{}.pickle'.format(dataset_eval), 'rb') as fh:
            meta = pickle.load(fh)
    except:
        meta = dict()
    combine_cats_iou = dict()


    
    for input_bbox_type in ['inmodal', 'amodal']:
        l = '\n==================================================\n==================================================\nKnown Center (k.c.): ' + input_bbox_type=='amodal' + '\n'
        print_(l, file=file)
        if TABLE_NUM == 1:
            print_('=========================\n{:25}{}\n{:25}{}\n\n'.format('FG_levels:', fg_levels, 'BG_levels:', bg_levels), file=file)

            for fg_level in fg_levels:

                eval_modes_ = copy.deepcopy(eval_modes)
                bg_levels_ = bg_levels
                if fg_level == 0:
                    bg_levels_ = [0]
                    if 'occ' in eval_modes_:
                        eval_modes_.remove('occ')

                for bg_level in bg_levels_:
                    sub_tag = 'FGL{}_BGL{}'.format(fg_level, bg_level)
                    print('========FGL{} BGL{}========'.format(fg_level, bg_level))
                    with open('../RPN_results/mrcnn_inmodal_RPN_FGL{}_BGL{}.pickle'.format(fg_level,
                                                                                                    bg_level),
                            'rb') as fh:
                        rpn_results = pickle.load(fh)

                    data_set = Occ_Veh_Dataset(cats=categories['eval'], dataType=dataType, train_types=[None], fg_level=fg_level,
                                            bg_level=bg_level, single_obj=True, resize=False, crop_img=False,
                                            crop_padding=48, data_range=[0, fraction_to_load], crop_central=False,
                                            demo_img_return=True)
                    data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False)

                    meta[sub_tag] = eval_performance(data_loader, rpn_results, category='all', demo=bool_demo_seg,
                                                    eval_modes=eval_modes_, input_bbox_type=input_bbox_type,
                                                    input_gt_label=bool_gt_label)

        elif TABLE_NUM == 2:
            print_('=========================\n{:25}{}\n{:25}{}\n\n'.format('Height_thrd:', height_thrd, 'Occ_bounds:', occ_bounds), file=file)

            for occ_bound in occ_bounds:
                occ_bound_str = '[{:.1f}, {:.1f}]'.format(occ_bound[0], occ_bound[1])
                eval_modes_ = copy.deepcopy(eval_modes)

                if occ_bound[1] == 0 and 'occ' in eval_modes_:
                    eval_modes_.remove('occ')

                print('========Height_thrd: {} - Occlusion_bound: {}========'.format(height_thrd, occ_bound_str))

                with open('../RPN_results/mrcnn_inmodal_kins_RPN.pickle', 'rb') as fh:
                    rpn_results = pickle.load(fh)

                data_set = KINS_Dataset(category_list=categories['eval'], dataType=dataType, occ=occ_bound, height_thrd=height_thrd,
                                        amodal_height=True, data_range=[0, fraction_to_load], demo_img_return=True)
                data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False)

                meta['kinsv'] = eval_performance(data_loader, rpn_results, category='kinsv', demo=bool_demo_seg, eval_modes=eval_modes_, input_bbox_type=input_bbox_type, input_gt_label=bool_gt_label, search_for_file=True)


        elif TABLE_NUM == 3:
            print_('=========================\n{:25}{}\n\n'.format('FG_levels:', fg_levels,), file=file)

            with open('../RPN_results/mrcnn_inmodal_RPN_cocoa.pickle', 'rb') as fh:
                rpn_results = pickle.load(fh)

            for fg_level in fg_levels:

                eval_modes_ = copy.deepcopy(eval_modes)
                bg_levels_ = bg_levels
                if fg_level == 0:
                    bg_levels_ = [0]
                    if 'occ' in eval_modes_:
                        eval_modes_.remove('occ')

                for bg_level in bg_levels_:

                    print('========FGL{}========'.format(fg_level))
                    for category in categories['eval']:
                        sub_tag = '{}FGL{}_BGL{}'.format(category, fg_level, bg_level)

                        data_set = COCOA_Dataset(cats=[category], dataTypes=['train', 'val'], fg_level=fg_level, resize=False, crop_img=False, crop_padding=48, data_range=[0, fraction_to_load], crop_central=False, demo_img_return=True)
                        data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False)

                        meta[sub_tag] = eval_performance(data_loader, rpn_results, category=category, demo=bool_demo_seg, eval_modes=eval_modes_, input_bbox_type=input_bbox_type, input_gt_label=bool_gt_label)

                    acc = 0
                    amodal_box_iou = 0
                    total = 0
                    for eval in eval_modes_:
                        combine_cats_iou[eval] = 0

                    for category in categories['eval']:
                        sub_tag = '{}FGL{}_BGL{}'.format(category, fg_level, bg_level)
                        num_obj = meta[sub_tag]['cls']['cls_acc'].shape[0]
                        if num_obj == 0:
                            continue

                        acc += meta[sub_tag]['cls']['cls_acc_val'] * num_obj
                        amodal_box_iou += meta[sub_tag]['cls']['amodal_bbox_iou_val'] * num_obj
                        total += num_obj

                        for eval in eval_modes_:
                            combine_cats_iou[eval] += meta[sub_tag][eval]['average iou'] * num_obj

                    level_tag = 'FGL{}_BGL{}'.format(fg_level, bg_level)
                    print_line = '{:15}  - {:5} -  cls_acc: {:6.4f}     |     amodal_box_mIoU: {:6.4f}     '.format(level_tag, total, acc / total, amodal_box_iou / total)
                    meta[level_tag] = {'cls_acc' : acc / total, 'amodal_box_mIoU' : amodal_box_iou / total}
                    for eval in eval_modes_:
                        print_line += '|     {}_mIoU: {:6.4f}     '.format(eval, combine_cats_iou[eval] / total)
                        meta[level_tag]['{}_mIoU'.format(eval)] = combine_cats_iou[eval] / total

                    print_('\n{}\n\n'.format(print_line), file=file)


    with open(overall_exp_dir + '/exp_meta.pickle'.format(dataset_eval), 'wb') as fh:
        pickle.dump(meta, fh)

    file.close()
