from Code.vMFMM import *
from config_initialization import vc_num, dataset, categories, data_path, cat_test, device_ids, Astride, Apad, Arf, vMF_kappa, layer,init_path, nn_type, dict_dir, offset, extractor
from Code.helpers import getImg, imgLoader, Imgset, myresize
from DataLoader import KINS_Compnet_Train_Dataset
import torch
from torch.utils.data import DataLoader
import cv2
import glob
import pickle
import os

kins_category = ['_', 'cyclist', 'pedestrian', '_', 'car', 'tram', 'truck', 'van', 'misc']

categories_to_train = ['cyclist', 'car', 'tram', 'truck', 'van']

img_per_cat = 3000
samp_size_per_img = 20
height_threshold = 75

for vMF_kappa in [50, 60]:
	loc_set = []
	feat_set = []
	imgs = []
	nfeats = 0

	for category in categories_to_train:
		print('{}'.format(category))
		cur_img_num = 0
		imgset = KINS_Compnet_Train_Dataset(category=category, height_thrd=height_threshold)
		data_loader = DataLoader(dataset=imgset, batch_size=1, shuffle=True)


		for ii,data in enumerate(data_loader):
			input, demo_img, img_path, true_pad = data
			imgs.append(img_path[0])

			img = demo_img[0].numpy()

			if input.shape[3] < 32:
				continue
			if np.mod(ii,50)==0:
				print('{} / {}'.format(ii,img_per_cat), end='\r')

			if cur_img_num >= img_per_cat:
				break

			with torch.no_grad():
				tmp = extractor(input.cuda(device_ids[0]))[0].detach().cpu().numpy()
			height, width = tmp.shape[1:3]

			# if np.random.random() > 0.95:
			# 	cv2.imwrite('temp.jpg', img)
			# 	assert False
			tmp = tmp[:,offset:height - offset, offset:width - offset]
			gtmp = tmp.reshape(tmp.shape[0], -1)
			if gtmp.shape[1] >= samp_size_per_img:
				rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img]
			else:
				rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img - gtmp.shape[1]]
				#rand_idx = np.append(range(gtmp.shape[1]), rand_idx)
			tmp_feats = gtmp[:, rand_idx].T

			cnt = 0
			for rr in rand_idx:
				ihi, iwi = np.unravel_index(rr, (height - 2 * offset, width - 2 * offset))
				hi = (ihi+offset)*(input.shape[2]/height)-Apad
				wi = (iwi + offset)*(input.shape[3]/width)-Apad

				loc_set.append([categories_to_train.index(category), ii, hi,wi,hi+Arf,wi+Arf])
				feat_set.append(tmp_feats[cnt,:])
				cnt+=1

			cur_img_num += 1
		print()


	feat_set = np.asarray(feat_set)
	loc_set = np.asarray(loc_set).T

	print(feat_set.shape)
	model = vMFMM(vc_num, 'k++')
	model.fit(feat_set, vMF_kappa, max_it=150)

	S = np.zeros((vc_num, vc_num))
	for i in range(vc_num):
		for j in range(i, vc_num):
			S[i,j] = np.dot(model.mu[i], model.mu[j])
	print('kap {} sim {}'.format(vMF_kappa,np.mean(S+S.T-np.diag(np.ones(vc_num)*2))))

	with open(dict_dir + 'dictionary_{}_{}_kappa{}.pickle'.format(layer,vc_num, vMF_kappa), 'wb') as fh:
		pickle.dump(model.mu, fh)
#
#
# num = 50
# SORTED_IDX = []
# SORTED_LOC = []
# for vc_i in range(vc_num):
# 	sort_idx = np.argsort(-model.p[:, vc_i])[0:num]
# 	SORTED_IDX.append(sort_idx)
# 	tmp=[]
# 	for idx in range(num):
# 		iloc = loc_set[:, sort_idx[idx]]
# 		tmp.append(iloc)
# 	SORTED_LOC.append(tmp)
#
# with open(dict_dir + 'dictionary_{}_{}_p.pickle'.format(layer,vc_num), 'wb') as fh:
# 	pickle.dump(model.p, fh)
# p = model.p
#
# print('save top {0} images for each cluster'.format(num))
# example = [None for vc_i in range(vc_num)]
# out_dir = dict_dir + '/cluster_images_{}_{}/'.format(layer,vc_num)
# if not os.path.exists(out_dir):
# 	os.makedirs(out_dir)
#
# print('')
#
# for vc_i in range(vc_num):
# 	patch_set = np.zeros(((Arf**2)*3, num)).astype('uint8')
# 	sort_idx = SORTED_IDX[vc_i]#np.argsort(-p[:,vc_i])[0:num]
# 	opath = out_dir + str(vc_i) + '/'
# 	if not os.path.exists(opath):
# 		os.makedirs(opath)
# 	locs=[]
# 	for idx in range(num):
# 		iloc = loc_set[:,sort_idx[idx]]
# 		category = iloc[0]
# 		loc = iloc[1:6].astype(int)
# 		if not loc[0] in locs:
# 			locs.append(loc[0])
# 			img = cv2.imread(imgs[int(loc[0])])
# 			img = myresize(img, 224, 'short')
# 			patch = img[loc[1]:loc[3], loc[2]:loc[4], :]
# 			#patch_set[:,idx] = patch.flatten()
# 			if patch.size:
# 				cv2.imwrite(opath+str(idx)+'.JPEG',patch)
# 	#example[vc_i] = np.copy(patch_set)
# 	if vc_i%10 == 0:
# 		print(vc_i)
#
# # print summary for each vc
# #if layer=='pool4' or layer =='last': # somehow the patches seem too big for p5
# for c in range(vc_num):
# 	iidir = out_dir + str(c) +'/'
# 	files = glob.glob(iidir+'*.JPEG')
# 	width = 100
# 	height = 100
# 	canvas = np.zeros((0,4*width,3))
# 	cnt = 0
# 	for jj in range(4):
# 		row = np.zeros((height,0,3))
# 		ii=0
# 		tries=0
# 		next=False
# 		for ii in range(4):
# 			if (jj*4+ii)< len(files):
# 				img_file = files[jj*4+ii]
# 				if os.path.exists(img_file):
# 					img = cv2.imread(img_file)
# 				img = cv2.resize(img, (width,height))
# 			else:
# 				img = np.zeros((height, width, 3))
# 			row = np.concatenate((row, img), axis=1)
# 		canvas = np.concatenate((canvas,row),axis=0)
#
# 	cv2.imwrite(out_dir+str(c)+'.JPEG',canvas)
