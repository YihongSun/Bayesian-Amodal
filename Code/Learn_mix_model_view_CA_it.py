from sklearn.cluster import SpectralClustering
from torch.utils.data import DataLoader
import gc

from configs import device_ids, dataset_train, dataset_eval, nn_type, vc_num, K, vMF_kappa, context_cluster, layer, meta_dir, categories, feature_num, feat_stride
from configs import *
from DataLoader import get_pascal3d_data, get_coco_data, Single_Object_Loader, KINS_Compnet_Train_Dataset
from util import roc_curve, rank_perf, visualize, res_down, graph_prior
from model import get_compnet_head, get_mixture_models

from old_Net_E2E import Conv1o1Layer
import torch

ITERATION = 3
LOW_RES =   False

def learn_mix_model_vMF(category, num_layers=2, num_clusters_per_layer=2, frac_data=1.0, iteration=0, CONTEXT_THRD=0.5):

    # Spectral clustering based on the similarity matrix

    sim_fname = meta_dir + 'init_vgg/' + 'similarity_vgg_pool4_{}/'.format(dataset_train) + 'simmat_mthrh045_{}_K{}.pickle'.format(category, 512)

    with open(sim_fname, 'rb') as fh:
        mat_dis1, _ = pickle.load(fh)

    mat_dis = mat_dis1
    subN = np.int(mat_dis.shape[0] * frac_data)
    mat_dis = mat_dis[:subN, :subN]
    print('total number of instances for obj {}: {}     context_thrd: {}'.format(category, subN, CONTEXT_THRD))
    N = subN
    img_idx = np.asarray([nn for nn in range(N)])

    #Dataloader
    if dataset_train == 'pascal3d+':
        image_files, mask_files, labels, bboxs = get_pascal3d_data(cats=[category], train=True, single_obj=True)
        data_set = Single_Object_Loader(image_files, mask_files, labels, bboxs, resize=True, crop_img=True, crop_padding=48, crop_central=True, demo_img_return=True, return_true_pad=True)
    if dataset_train == 'kinsv':
        data_set = KINS_Compnet_Train_Dataset(category=category, height_thrd=75, pad=48)

    data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False)

    r_set_fg = [None for nn in range(N)]
    r_set_context = [None for nn in range(N)]
    img_set = [None for nn in range(N)]

    for ii, data in enumerate(data_loader):
        if dataset_train == 'pascal3d+':
            input, label, bbox, gt_mask, demo_img, img_path, true_pad = data
        if dataset_train == 'kinsv':
            input, demo_img, img_path, true_pad = data

        img = demo_img.numpy().squeeze()
        true_pad = true_pad.numpy().squeeze()

        input.requires_grad = False
        vc_activation, fg_mask = net.get_vc_activation_with_binary_fg_mask(input.cuda(device_ids[0]), gt_category=categories['train'].index(category), use_context_center=(iteration == 0), use_mixture_model=(iteration > 0), context_thrd=CONTEXT_THRD, mmodel_thrd=0, bmask_post_process=True, cntxt_pad=true_pad // feat_stride, low_res=LOW_RES)

        context_mask = 1 - fg_mask

        fg_map = vc_activation * np.repeat(fg_mask[np.newaxis, :, :], vc_num, axis=0)
        context_map = vc_activation * np.repeat(context_mask[np.newaxis, :, :], vc_num, axis=0)

        r_set_fg[ii] = fg_map.transpose((0, 2, 1))
        r_set_context[ii] = context_map.transpose((0, 2, 1))
        img_set[ii] = img

        if ii % 10 == 0:
            print('{}/{}'.format(ii, N), end='\r')
        if ii == N - 1:
            break

        # visualize(img, fg_mask, 'temp')
        # assert False


    # num cluster centers
    max_0 = vc_num  # max([layer_feature[nn].shape[2] for nn in range(N)])
    # width
    max_1 = max([r_set_fg[nn].shape[1] for nn in range(N)])
    # height
    max_2 = max([r_set_fg[nn].shape[2] for nn in range(N)])
    print(max_0, max_1, max_2)

    layer_feature_vmf = np.zeros((N, max_0, max_1, max_2), dtype=np.float32)
    layer_feature_vmf_context = np.zeros((N, max_0, max_1, max_2), dtype=np.float32)        ##CONTEXT AWARE

    for nn in range(N):
        # try:
        vnum, ww, hh = r_set_fg[nn].shape
        # except:
        #	print('')
        assert (vnum == max_0)
        diff_w1 = int((max_1 - ww) / 2)
        diff_w2 = int(max_1 - ww - diff_w1)
        assert (max_1 == diff_w1 + diff_w2 + ww)
        diff_h1 = int((max_2 - hh) / 2)
        diff_h2 = int(max_2 - hh - diff_h1)
        assert (max_2 == diff_h1 + diff_h2 + hh)
        padded_fg = np.pad(r_set_fg[nn], ((0, 0), (diff_w1, diff_w2), (diff_h1, diff_h2)), 'constant', constant_values=0)
        padded_context = np.pad(r_set_context[nn], ((0, 0), (diff_w1, diff_w2), (diff_h1, diff_h2)), 'constant', constant_values=0)

        layer_feature_vmf[nn, :, :, :] = padded_fg
        layer_feature_vmf_context[nn, :, :, :] = padded_context

        r_set_fg[nn] = []
        r_set_context[nn] = []



    mat_full = mat_dis + mat_dis.T - np.ones((subN, subN))
    np.fill_diagonal(mat_full, 0)

    # W_mat = 1. - mat_full
    # print('W_mat stats: {}, {}'.format(np.mean(W_mat), np.std(W_mat)))
    mat_sim = 1. - mat_full

    K = np.power(num_clusters_per_layer, num_layers)
    view_dir = os.path.join(mixdir, 'viewpoints_K{}_FEATDIM{}_{}/'.format(K, vc_num, layer))
    set_type = 'train'
    # dataset_root = '/home/user/projects/vc/data/PASCAL3D+/'
    '''
    dataset_root = data_dir + 'PASCAL3D+/'
    list_dir = os.path.join(dataset_root, 'PASCAL3D+_release1.1', 'Image_sets')
    anno_dir = os.path.join(dataset_root, 'PASCAL3D+_release1.1', 'Annotations_mat', '{}_imagenet'.format(category))        #Annotations
    import scipy.io as sio

    # Load viewpoint ground truth viewpoint data
    if not os.path.exists(view_dir):
        os.mkdir(view_dir)
    # cache_dir = os.path.join(proj_root, 'feat')
    if set_type == 'train':
        filelist = os.path.join(list_dir, '{}_imagenet_train.txt').format(category)
    else:
        filelist = os.path.join(list_dir, '{}_imagenet_val.txt').format(category)
    with open(filelist, 'r') as fh:
        contents = fh.readlines()
    '''
    view_point = []
    '''
    for cc in contents:
        mat_file = os.path.join(anno_dir, '{}.mat'.format(cc.strip()))
        assert (os.path.isfile(mat_file))
        mat_contents = sio.loadmat(mat_file)
        record = mat_contents['record']
        objects = record['objects']
        view_point.append(objects[0, 0]['viewpoint'][0, 0]['azimuth_coarse'][0, 0][0, 0])

    view_point = view_point[0:N]
    assert (N == len(view_point))
    for rr in np.random.randint(N, size=10):
        ref = rr
        # print(view_point[ref])
        sim_rst = mat_sim[rr]
        max_idx = np.argsort(-sim_rst)
        for tt in range(5):
            match = max_idx[tt]
    #	print(view_point[match], end=' ')

    # print('.')
    '''

    # setup caching variables
    tmp = list()
    tmp.append(np.zeros(mat_sim.shape[0]))
    LABELS = list()
    LABELS.append(tmp)
    tmp = list()
    tmp.append(mat_sim)
    MAT = list()
    MAT.append(tmp)
    tmp = list()
    tmp.append(view_point)
    VIEW = list()
    VIEW.append(tmp)
    tmp = list()
    tmp.append(range(N))
    IMAGEIDX = list()
    IMAGEIDX.append(tmp)

    import matplotlib.pyplot as plt
    # plot variables
    nbins = 32
    bottom = 0
    max_height = 4
    theta = np.linspace(0.0, 2 * np.pi, nbins, endpoint=False)
    width = (2 * np.pi) / nbins

    viewpoint_images = []
    # start clustering
    for i in range(num_layers):
        MAT_SUB = list()
        LABELS_SUB = list()
        VIEW_SUB = list()
        IMAGEIDX_SUB = list()

        print('Clustering layer {} ...'.format(i))
        for k in range(np.power(num_clusters_per_layer, i)):
            parent_counter = int(np.floor(k / num_clusters_per_layer))
            leaf_counter = int(np.mod(k, num_clusters_per_layer))
            idx = np.where(LABELS[i][parent_counter] == leaf_counter)[0]
            mat_sim_sub = MAT[i][parent_counter][np.ix_(idx, idx)]  # subsample similarity matrix
            MAT_SUB.append(mat_sim_sub)
            '''
            view_sub = np.array(VIEW[i][parent_counter])[idx]  # subsample view
            VIEW_SUB.append(view_sub)
            '''
            IMAGEIDX_SUB.append(np.array(IMAGEIDX[i][parent_counter])[idx])
            # print(np.mean(mat_sim_sub[k]), np.std(mat_sim_sub[k]))

            cls_solver = SpectralClustering(n_clusters=num_clusters_per_layer, affinity='precomputed', random_state=0)
            cluster_result = cls_solver.fit_predict(mat_sim_sub)
            LABELS_SUB.append(cluster_result)

            # plot
            '''
			f, axes = plt.subplots(1,num_clusters_per_layer, sharex=True, sharey=True, figsize=(8,4))
			axes[0].set_xlim([0,360])
			axes[0].set_xticks([0,90,180,270,360])
			for kk in range(num_clusters_per_layer):
				axes[kk].hist(view_sub[cluster_result==kk], bins=16, range=[0,360], facecolor='green', alpha=0.75,edgecolor='k')
			plt.close()
			'''
            '''
            plt.figure(figsize=(5, 8))
            for kk in range(num_clusters_per_layer):
                radii, _ = np.histogram(np.array(view_sub)[cluster_result == kk], bins=nbins, range=[0, 360])
                # radii = max_height*(radii/np.max(radii))
                ax = plt.subplot(num_clusters_per_layer, 1, kk + 1, polar=True)
                bars = ax.bar(theta, radii, width=width, bottom=bottom)
                # Use custom colors and opacity
                for r, bar in zip(radii, bars):
                    bar.set_facecolor(plt.cm.jet(r / (N * 0.40 / K)))
                # bar.set_alpha(0.8)
            save_name = view_dir + '{}_{}_1000_vp_layer-{}_node-{}.png'.format(category, set_type, i, k)
            plt.savefig(save_name, bbox_inches='tight')
            plt.close()
            if i == num_layers - 1:
                tmp = cv2.imread(save_name)
                # split images into inidivual plots
                fsz = int(tmp.shape[0] / num_clusters_per_layer)
                for zz in range(num_clusters_per_layer):
                    subfig = tmp[zz * fsz:(zz + 1) * fsz, :, :]
                    viewpoint_images.append(subfig)
            '''

        MAT.append(MAT_SUB)
        LABELS.append(LABELS_SUB)
        '''
        VIEW.append(VIEW_SUB)
        '''
        IMAGEIDX.append(IMAGEIDX_SUB)

    mixmodel_lbs = np.ones(len(LABELS[0][0])) * -1
    for i in range(len(LABELS[-1])):
        lab_sub = LABELS[-1][i]
        for j in range(len(lab_sub)):
            mixmodel_lbs[IMAGEIDX[-1][i][j]] = (i * num_clusters_per_layer) + lab_sub[j]

    print('')


    mixmodel_lbs = mixmodel_lbs[:N]

    for kk in range(K):
        print('cluster {} has {} samples'.format(kk, np.sum(mixmodel_lbs == kk)))
    # assert(subN == np.sum(all_N))
    alpha = []      ##FOREGROUND
    beta = []       ##CONTEXT
    prior = []      #prob on fg or bg

    for kk in range(K):
        # get samples for mixture component
        bool_clust = mixmodel_lbs == kk
        bidx = [i for i, x in enumerate(bool_clust) if x]
        num_clusters = vc_num  # vmf.shape[1]
        # loop over samples
        for idx in bidx:
            # compute
            vmf_sum = np.sum(layer_feature_vmf[img_idx[idx]], axis=0)
            vmf_sum = np.reshape(vmf_sum, (1, vmf_sum.shape[0], vmf_sum.shape[1]))
            vmf_sum = vmf_sum.repeat(num_clusters, axis=0) + 1e-3
            mask = vmf_sum > 0
            layer_feature_vmf[img_idx[idx]] = mask * (layer_feature_vmf[img_idx[idx]] / vmf_sum)

        N_samp = np.sum(layer_feature_vmf[img_idx[bidx]] > 0, axis=0)  # stores the number of samples
        fg_mask = np.sum(np.sum(layer_feature_vmf[img_idx[bidx]], axis=0).T, axis=2) / len(bidx)

        mask = (N_samp > 0)
        vmf_sum = mask * (np.sum(layer_feature_vmf[img_idx[bidx]], axis=0) / (N_samp + 1e-5)).astype(np.float32)
        alpha.append(vmf_sum)

        for idx in bidx:
            # compute
            vmf_sum = np.sum(layer_feature_vmf_context[img_idx[idx]], axis=0)
            vmf_sum = np.reshape(vmf_sum, (1, vmf_sum.shape[0], vmf_sum.shape[1]))
            vmf_sum = vmf_sum.repeat(num_clusters, axis=0) + 1e-3
            mask = vmf_sum > 0
            layer_feature_vmf_context[img_idx[idx]] = mask * (layer_feature_vmf_context[img_idx[idx]] / vmf_sum)

        N_samp = np.sum(layer_feature_vmf_context[img_idx[bidx]] > 0, axis=0)  # stores the number of samples
        context_mask = np.sum(np.sum(layer_feature_vmf_context[img_idx[bidx]], axis=0).T, axis=2) / len(bidx)
        mask = (N_samp > 0)
        vmf_sum = mask * (np.sum(layer_feature_vmf_context[img_idx[bidx]], axis=0) / (N_samp + 1e-5)).astype(np.float32)
        beta.append(vmf_sum)

        prior.append([fg_mask, context_mask])

        gc.collect()
    '''
	# ML updates of mixture model and vMF mixture coefficients
	'''
    W_mat = None
    W_mat2 = None
    mat_dis = None
    mat_dis1 = None
    mat_full = None
    vmF = None
    numsteps = 10

    for ee in range(numsteps):
        changed = 0
        mixture_likeli = np.zeros((subN, K))
        print('\nML Step {} / {}'.format(ee, numsteps))
        changed_samples = np.zeros(subN)
        for nn in range(subN):
            if nn % 100 == 0:
                print('{}'.format(nn))
            # compute feature likelihood
            for kk in range(K):
                like_map = layer_feature_vmf[img_idx[nn]] * alpha[kk]
                likeli = np.sum(like_map, axis=0) + 1e-10
                mixture_likeli[nn, kk] = np.sum(np.log(likeli))

            # compute new mixture assigment for feature map
            # sum_like=np.sum(like[nn,:])
            # like[nn,:] = like[nn,:] /sum_like
            new_assignment = np.argmax(mixture_likeli[nn, :])
            if new_assignment != mixmodel_lbs[nn]:
                changed += 1
                changed_samples[nn] = 1
            mixmodel_lbs[nn] = new_assignment

        for kk in range(K):
            print('cluster {} has {} samples'.format(kk, np.sum(mixmodel_lbs == kk)))
        print('{} changed assignments'.format(changed))

        # update mixture coefficients here
        for kk in range(K):
            # get samples for mixture component
            bool_clust = mixmodel_lbs == kk
            bidx = [i for i, x in enumerate(bool_clust) if x]
            num_clusters = vc_num  # vmf.shape[1]
            # loop over samples
            for idx in bidx:
                # compute
                vmf_sum = np.sum(layer_feature_vmf[img_idx[idx]], axis=0)
                vmf_sum = np.reshape(vmf_sum, (1, vmf_sum.shape[0], vmf_sum.shape[1]))
                vmf_sum = vmf_sum.repeat(num_clusters, axis=0) + 1e-3
                mask = vmf_sum > 0
                layer_feature_vmf[img_idx[idx]] = mask * (layer_feature_vmf[img_idx[idx]] / vmf_sum)

            N_samp = np.sum(layer_feature_vmf[img_idx[bidx]] > 0, axis=0)  # stores the number of samples
            fg_mask = np.sum(np.sum(layer_feature_vmf[img_idx[bidx]], axis=0).T, axis=2) / len(bidx)

            mask = (N_samp > 0)
            alpha[kk] = mask * (np.sum(layer_feature_vmf[img_idx[bidx]], axis=0) / (N_samp + 1e-5)).astype(np.float32)

            for idx in bidx:
                # compute
                vmf_sum = np.sum(layer_feature_vmf_context[img_idx[idx]], axis=0)
                vmf_sum = np.reshape(vmf_sum, (1, vmf_sum.shape[0], vmf_sum.shape[1]))
                vmf_sum = vmf_sum.repeat(num_clusters, axis=0) + 1e-3
                mask = vmf_sum > 0
                layer_feature_vmf_context[img_idx[idx]] = mask * (layer_feature_vmf_context[img_idx[idx]] / vmf_sum)

            N_samp = np.sum(layer_feature_vmf_context[img_idx[bidx]] > 0, axis=0)  # stores the number of samples
            context_mask = np.sum(np.sum(layer_feature_vmf_context[img_idx[bidx]], axis=0).T, axis=2) / len(bidx)
            mask = (N_samp > 0)
            beta[kk] = mask * (np.sum(layer_feature_vmf_context[img_idx[bidx]], axis=0) / (N_samp + 1e-5)).astype(
                np.float32)

            prior[kk] = [fg_mask, context_mask]

            gc.collect()

        print('')
        if changed / subN < 0.01:  # 25:
            break

    print('')
    '''
	# write images of clusters
	'''
    clust_img_dir = os.path.join(mixdir, 'clusters_K{}_FEATDIM{}_{}_specific_view/'.format(K, vc_num, layer))
    if not os.path.exists(clust_img_dir):
        os.makedirs(clust_img_dir)
    for kk in range(K):
        # img_ids_kk = np.where(img_idx[mixmodel_lbs == kk])[0]
        img_ids_kk = img_idx[mixmodel_lbs == kk]
        width = 300
        height = 150
        canvas = np.zeros((0, 4 * width, 3))
        cnt = 0
        for jj in range(4):
            row = np.zeros((height, 0, 3))
            for ii in range(4):
                if cnt < len(img_ids_kk):
                    img = img_set[img_ids_kk[cnt]]
                else:
                    img = np.zeros((height, width, 3))
                if not (dataset_train == 'mnist' or dataset_train == 'cifar10' or dataset_train == 'coco'):
                    img = cv2.resize(img, (width, height))
                row = np.concatenate((row, img), axis=1)
                cnt += 1
            canvas = np.concatenate((canvas, row), axis=0)
        cv2.imwrite(clust_img_dir + category + '_{}.JPEG'.format(kk), canvas)
        '''cv2.imwrite(clust_img_dir + category + '_{}_view.JPEG'.format(kk), viewpoint_images[kk])'''
        print('')

    prior_dir = os.path.join(mixdir, 'context_aware_prior_K{}_FEATDIM{}_{}/'.format(K, vc_num, layer))
    if not os.path.exists(prior_dir):
        os.makedirs(prior_dir)

    for kk in range(K):

        prob_map_fg = np.array(prior[kk][0])
        prob_map_bg = np.array(prior[kk][1])

        graph_prior(prob_map_fg, prob_map_bg, prior_dir + '{}_{}'.format(category, kk))

    savename = os.path.join(mixdir, 'mmodel_{}_K{}_FEATDIM{}_{}_specific_view_{}.pickle'.format(category, K, vc_num, layer, context_cluster))
    with open(savename, 'wb') as fh:
        pickle.dump([alpha, beta, prior], fh)


if __name__ == '__main__':

    model_tag = 'bayesian'

    net = get_compnet_head(mix_model_dim_reduction=False, mix_model_suffix='WILL_CAUSE_ERROR')

    # =================== Load up new vcs
    new_vcs = torch.load('new_vcs_06-09-2021-23-37.pt')
    net.vc_conv1o1 = Conv1o1Layer(new_vcs)
    print('New VCs loaded.')
    # =================== Load up new vcs

    try:
        with open(init_dir + 'context_thrds_{}/context_thrd_5.pickle'.format(dataset_train), 'rb') as fh:
            context_thrds = pickle.load(fh)
    except:
        context_thrds = dict()
        for category in categories['train']:
            context_thrds[category] = 0.54


    for it in range(ITERATION):

        mixdir = init_dir + 'mix_model_vmf_{}_EM_all_context_{}_it{}/'.format(dataset_train, model_tag, it)
        if not os.path.exists(mixdir):
            os.makedirs(mixdir)

        if it == 0:
            print('First Iteration: No Model Loading.')
        else:
            print('Load Model from iteration {}...'.format(it - 1))
            net.update_mixture_models(get_mixture_models(dim_reduction=False, tag='_{}_it{}'.format(model_tag, it - 1)))
            net.update_fused_models(omega=0.2)

        for category in categories['train']:        #categories['train']:

            for num_layers in [3]:
                learn_mix_model_vMF(category, num_layers=num_layers, num_clusters_per_layer=2, iteration=it, CONTEXT_THRD=context_thrds[category])
