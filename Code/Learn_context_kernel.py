from configs import init_dir, context_cluster, categories, dataset_train
from configs import *
from DataLoader import get_pascal3d_data, Multi_Object_Loader, KINS_Dataset
from model import get_backbone_extractor
from vMFMM import vMFMM
from torch.utils.data import DataLoader
from util import visualize
import random as rm

'''
Learn class specific contextual features
Code Status - Currently Active 6/26
'''


# this method restrict the sampling region to between the inner and outer bound
# Applied L2 normalization
def mask_features(features, bbox, img_shape, inner_bound, outer_bound):
    h, w, c = features.shape

    inner_bbox = bbox.copy()
    inner_bbox[:, 0] = (bbox[:, 0] - inner_bound) / img_shape[0] * h
    inner_bbox[:, 1] = (bbox[:, 1] - inner_bound) / img_shape[1] * w
    inner_bbox[:, 2] = (bbox[:, 2] + inner_bound) / img_shape[0] * h
    inner_bbox[:, 3] = (bbox[:, 3] + inner_bound) / img_shape[1] * w

    outer_bbox = bbox.copy()
    outer_bbox[:, 0] = (bbox[:, 0] - outer_bound) / img_shape[0] * h
    outer_bbox[:, 1] = (bbox[:, 1] - outer_bound) / img_shape[1] * w
    outer_bbox[:, 2] = (bbox[:, 2] + outer_bound) / img_shape[0] * h
    outer_bbox[:, 3] = (bbox[:, 3] + outer_bound) / img_shape[1] * w

    context_feat = []
    #demo = np.zeros((h, w))

    for i in range(h):
        for j in range(w):

            within_inner_bbox = False
            within_outer_bbox = False

            for b in range(bbox.shape[0]):

                inner_box = inner_bbox[b]
                within_inner_bbox = within_inner_bbox or (i >= inner_box[0] and i < inner_box[2] and j >= inner_box[1] and j < inner_box[3])

                outer_box = outer_bbox[b]
                within_outer_bbox = within_outer_bbox or (i >= outer_box[0] and i < outer_box[2] and j >= outer_box[1] and j < outer_box[3])

            if within_outer_bbox and not within_inner_bbox:
                context_feat.append((features[i][j] / np.sqrt(np.sum(features[i][j] ** 2) + 1e-10)).tolist())
                #demo[i][j] = 1
    return context_feat#, demo


def learn_context_feature(category, inner_bound=16, outer_bound=128, percentage_for_clustering=.1, max_num=100000, num_cluster=10):

    # Stage 1: Collect features that have receptive field outside of object bounding boxes
    print('==========Class: {}=========='.format(category))
    print('Stage 1: Feature Extraction')
    storage_dir = init_dir + 'context_features_meta_{}/'.format(dataset_train)
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)

    storage_file = storage_dir + '{}_context_features_[{} - {}].pickle'.format(category, inner_bound, outer_bound)

    if not os.path.exists(storage_file):

        feat_vec = []

        if dataset_train == 'pascal3d+':

            image_files, mask_files, labels, bboxs = get_pascal3d_data(cats=[category], train=True, single_obj=False)
            data_set = Multi_Object_Loader(image_files, mask_files, labels, bboxs, resize=True, min_size=200, max_size=3000, demo_img_return=True)

        elif dataset_train == 'kins':
            data_set = KINS_Dataset(category_list=[category], dataType='train', occ=[0, 1],
                                    height_thrd=75, amodal_height=False, frac=1.0,
                                    demo_img_return=True)

        data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False)

        N = data_loader.__len__()

        for ii, data in enumerate(data_loader):
            if dataset_train == 'pascal3d+':
                input, label, bbox, gt_mask, scale, demo_img, img_path = data

            elif dataset_train == 'kins':
                input, gt_labels, bbox, gt_amodal_bbox, gt_inmodal_segmentation, gt_amodal_segmentation, gt_occ, demo_img, img_path = data
            bbox = bbox.numpy()

            if input.shape[2] * input.shape[3] > 2000 * 2000:
                continue

            bbox = bbox[0]
            layer_feature = extractor(input.cuda(device_ids[0]))[0].detach().cpu().numpy()
            layer_feature = layer_feature.transpose((1, 2, 0))

            # the sampling region must be the region within 'bounding box pad with outer_bound' and outside of 'bounding box pad with inner_bound'
            # L2 normalization is applied within the mask_features
            feat_vec = feat_vec + mask_features(layer_feature, bbox, input.shape[2:4], inner_bound, outer_bound)

            if ii % 10 == 0:
                print("{}/{}      Collected:     {}          ".format(ii, N, len(feat_vec)), end='\r')
                if ii == 2000:
                    break

        print()
        with open(storage_file, 'wb') as fh:
            pickle.dump(feat_vec, fh)
    else:
        with open(storage_file, 'rb') as fh:
            feat_vec = pickle.load(fh)

    print('         Complete  --  {} features found'.format(len(feat_vec)))
    print()
    print()

    print('Stage 2: Feature Clustering')
    rm.seed(0)
    X = np.array(feat_vec)
    rand_idx = rm.sample(range(X.shape[0]), min((int)(percentage_for_clustering * X.shape[0]), max_num))
    X = X[rand_idx]
    feat_vec = None

    print('         Training Shape:', X.shape)

    model = vMFMM(num_cluster, 'k++')  # cluster_num_class_specific = 50
    model.fit(X, 30, max_it=150, tol=5e-6)
    context_centers = model.mu

    for i in range(context_centers.shape[0]):
        context_centers[i] = context_centers[i] / np.sqrt(np.sum(context_centers[i] ** 2) + 1e-10)


    if X.shape[0] > 1000:
        rand_idx_2 = rm.sample(range(X.shape[0]), 1000)
    else:
        rand_idx_2 = range(X.shape[0])

    perf = []
    for feat in X[rand_idx_2]:
        match = 0
        for center in context_centers:
            if np.sum(feat * center) > 0.50:
                match = 1
                break
        perf.append(match)

    print('         Complete  --  Feature Population Coverage: {}%'.format(np.mean(perf) * 100))

    print()
    print()

    X = None

    print('Stage 3: Making demo...')

    feature_collection = []
    img_collection = []


    if dataset_train == 'pascal3d+':
        image_files, mask_files, labels, bboxs = get_pascal3d_data(cats=[category], train=False, single_obj=False)
        data_set = Multi_Object_Loader(image_files, mask_files, labels, bboxs, resize=True, min_size=200, max_size=3000,
                                       demo_img_return=True)

    elif dataset_train == 'kins':
        data_set = KINS_Dataset(category_list=[category], dataType='test', occ=[0, 1],
                                height_thrd=75, amodal_height=False, frac=1.0,
                                demo_img_return=True)

    data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=True)

    # go through 100 images for material
    for ii, data in enumerate(data_loader):
        if dataset_train == 'pascal3d+':
            input, label, bbox, gt_mask, scale, demo_img, img_path = data

        elif dataset_train == 'kins':
            input, gt_labels, bbox, gt_amodal_bbox, gt_inmodal_segmentation, gt_amodal_segmentation, gt_occ, demo_img, img_path = data

        img = demo_img.numpy().squeeze()

        if input.shape[2] * input.shape[3] > 2000 * 2000:
            continue

        layer_feature = extractor(input.cuda(device_ids[0]))[0].detach().cpu().numpy()
        iheight, iwidth = layer_feature.shape[1:3]
        lff = layer_feature.reshape(layer_feature.shape[0], -1).T
        lff_norm = lff / (np.sqrt(np.sum(lff ** 2, 1) + 1e-10).reshape(-1, 1)) + 1e-10

        lff_norm = lff_norm.reshape(iheight, iwidth, -1).astype(np.float32).T
        feat_norm = lff_norm.transpose((2, 1, 0))

        feature_collection.append(feat_norm)
        img_collection.append(img)

        if ii > 100:
            break

    # go through all the contexts
    for context_center_ii in range(context_centers.shape[0]):
        print('         Center {}/{}'.format(context_center_ii, context_centers.shape[0]), end='\r')

        context_center = context_centers[context_center_ii]

        max_val = []
        context_activation_collection = []

        for i in range(len(feature_collection)):
            activation = feature_collection[i]
            # context_activation = np.log(np.sum(activation * context_center, axis=2) + 1e-6)
            context_activation = np.sum(activation * context_center, axis=2)
            context_activation_collection.append(context_activation)
            max_val.append(np.max(context_activation))

        rank_ind = np.argsort(max_val)[::-1]

        # making images
        demo_kernel_dir = storage_dir + '{}_kernel_demo_{}/'.format(category, num_cluster)
        if not os.path.exists(demo_kernel_dir):
            os.makedirs(demo_kernel_dir)
        rf_size = 16
        demo_size = 100
        canvas = np.zeros((demo_size * 4, demo_size * 4, 3))
        for r in range(16):
            img = img_collection[rank_ind[r]]
            context_act = context_activation_collection[rank_ind[r]]

            row_num = int(r / 4)
            col_num = r % 4

            ind = np.unravel_index(np.argmax(context_activation_collection[rank_ind[r]], axis=None),
                                   context_activation_collection[rank_ind[r]].shape)
            x = int(ind[0] / context_act.shape[0] * img.shape[0])
            y = int(ind[1] / context_act.shape[1] * img.shape[1])
            img_patch = img[x:min(x + rf_size, img.shape[0]), y:min(y + rf_size, img.shape[1]), :]

            canvas[row_num * demo_size: (row_num + 1) * demo_size, col_num * demo_size: (col_num + 1) * demo_size,
            :] = cv2.resize(img_patch, (demo_size, demo_size))

            if r < 5:
                visualize(img, context_act, demo_kernel_dir + 'kernel_demo_{}_{}_{}'.format(category, context_center_ii, r),
                          cbar=True)

        cv2.imwrite(demo_kernel_dir + '{}_context_center_{}.jpg'.format(category, context_center_ii), canvas)

    print()
    print()

    final_dir = init_dir + 'context_kernel_{}_{}/'.format(layer, dataset_train)
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
    final_file = final_dir + '{}_{}.npy'.format(category, num_cluster)
    np.save(final_file, context_centers)


if __name__ == '__main__':

    extractor = get_backbone_extractor()
    print('Number of Context Features Per Category:', context_cluster)

    for category in categories['train']:            # ['bicycle','car']:
        learn_context_feature(category, inner_bound=32, outer_bound=128, num_cluster=context_cluster)
