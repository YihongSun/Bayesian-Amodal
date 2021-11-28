from configs import device_ids, dataset_train, dataset_eval, nn_type, vc_num, K, vMF_kappa, context_cluster, layer, exp_dir, categories, feature_num
from configs import *
import random
import torch
import torch.nn as nn


class ClusterLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, y):
		loss = 0
		y = y.squeeze(3).squeeze(2)
		for i in x:
			i = i.permute(1, 2, 0)
			i = i.reshape(-1, i.shape[2])
			length = i.shape[0]

			m = i / i.norm(dim=1, keepdim=True)
			n = (y / y.norm(dim=1, keepdim=True)).transpose(0, 1)

			z = m.mm(n)
			z = z.max(dim=1)[0]
			loss += (1. - z).sum() / float(length)
		return loss

class WeaklySupMaskLoss(nn.Module):
	def __init__(self, cls_loss, mil_coef=1, pairwise_coef=0.05, sampling_fraction=0.2):
		super().__init__()

		self.cls_loss = cls_loss
		self.mil_coef = mil_coef
		self.pairwise_coef = pairwise_coef
		self.sampling_fraction = sampling_fraction

		self.center_weight = torch.tensor([[0., 0., 0.], [0., 1., 0.],
									  [0., 0., 0.]])  # , device=device)

		self.pairwise_weights_list = [
			torch.tensor([[0., 0., 0.], [1., 0., 0.],
						  [0., 0., 0.]]),  # , device=device),
			torch.tensor([[0., 0., 0.], [0., 0., 1.],
						  [0., 0., 0.]]),  # , device=device),
			torch.tensor([[0., 1., 0.], [0., 0., 0.],
						  [0., 0., 0.]]),  # , device=device),
			torch.tensor([[0., 0., 0.], [0., 0., 0.],
						  [0., 1., 0.]]),  # , device=device),
			torch.tensor([[1., 0., 0.], [0., 0., 0.],
						  [0., 0., 0.]]),  # , device=device),
			torch.tensor([[0., 0., 1.], [0., 0., 0.],
						  [0., 0., 0.]]),  # , device=device),
			torch.tensor([[0., 0., 0.], [0., 0., 0.],
						  [1., 0., 0.]]),  # , device=device),
			torch.tensor([[0., 0., 0.], [0., 0., 0.],
						  [0., 0., 1.]]),  # , device=device),
		]

	def forward(self, mask_prediction, bbox):
		h, w = bbox[2] - bbox[0], bbox[3] - bbox[1]

		factor = 224. / float(h)
		h, w = int(factor * h), int(factor * w)

		h_, w_ = int((mask_prediction.shape[2] - h) / 2), int((mask_prediction.shape[3] - w) / 2)

		targets = []
		labels = []

		targets_, labels_ = self.create_targets(mask_prediction[0, :, h_:h_ + h, w_:w_ + w].permute(1, 2, 0), label=1)
		targets += targets_
		labels += labels_

		if h_ > 1:
			targets_, labels_ = self.create_targets(mask_prediction[0, :, 0:h_, w_:w_ + w].permute(1, 2, 0), label=0, vertical_bands=False)
			targets += targets_
			labels += labels_

			targets_, labels_ = self.create_targets(mask_prediction[0, :, h_ + h:, w_:w_ + w].permute(1, 2, 0), label=0, vertical_bands=False)
			targets += targets_
			labels += labels_

		if w_ > 1:
			targets_, labels_ = self.create_targets(mask_prediction[0, :, h_:h_ + h, 0:w_].permute(1, 2, 0), label=0, horizontal_bands=False)
			targets += targets_
			labels += labels_

			targets_, labels_ = self.create_targets(mask_prediction[0, :, h_:h_ + h, w_ + w:].permute(1, 2, 0), label=0, horizontal_bands=False)
			targets += targets_
			labels += labels_

		rand_ind = random.sample(range(len(targets)), int(self.sampling_fraction * len(targets)))
		targets = [targets[i] for i in rand_ind]
		labels = [labels[i] for i in rand_ind]

		targets = torch.stack(targets)
		labels = torch.stack(labels).cuda(device_ids[0])

		pairwise_loss = []
		for w in self.pairwise_weights_list:
			conv = torch.nn.Conv2d(1, 1, 3, bias=False, padding=(1, 1))
			weights = self.center_weight - w
			weights = weights.view(1, 1, 3, 3).to(device_ids[0])
			conv.weight = torch.nn.Parameter(weights)
			for param in conv.parameters():
				param.requires_grad = False
			aff_map = conv(mask_prediction[:, 1:, :, :])

			cur_loss = (aff_map ** 2)
			cur_loss = torch.mean(cur_loss)
			pairwise_loss.append(cur_loss)

		return self.mil_coef * self.cls_loss(targets, labels[:, 0]) / targets.shape[0] + self.pairwise_coef * torch.mean(torch.stack(pairwise_loss))

	def create_targets(self, mask_prediction, label, horizontal_bands=True, vertical_bands=True):
		targets = []
		labels = []
		if horizontal_bands:
			for i in range(mask_prediction.shape[0]):
				max_ind = mask_prediction[i, :, 1].argmax()
				targets.append(mask_prediction[i, max_ind])
				labels.append(torch.Tensor([label]).type(torch.LongTensor))
		if vertical_bands:
			for j in range(mask_prediction.shape[1]):
				max_ind = mask_prediction[:, j, 1].argmax()
				targets.append(mask_prediction[max_ind, j])
				labels.append(torch.Tensor([label]).type(torch.LongTensor))
		return targets, labels




