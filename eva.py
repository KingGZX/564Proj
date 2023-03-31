import numpy as np
import scipy.stats
import os
import scipy.io
import torch
import glob
import h5py
from utils import *
from helper_s3dis import Data_Configs, Data_S3DIS


def eva_accu(model_path, data: Data_S3DIS, device):
	"""
	accuracy of semantic: backbone + seg_net

	"""
	backbone = torch.load(model_path["backbone"], map_location=torch.device('cpu'))
	backbone.eval()
	seg_net = torch.load(model_path["seg_net"], map_location=torch.device('cpu'))
	seg_net.eval()
	batch = data.total_test_batch_num

	TP_FP_Total = {}
	for sem_id in Data_Configs.sem_ids:
		TP_FP_Total[sem_id] = {}
		TP_FP_Total[sem_id]['TP'] = 0
		TP_FP_Total[sem_id]['FP'] = 0
		TP_FP_Total[sem_id]['Total'] = 0

	for i in range(batch):
		X, sem_labels, ins_labels, psem_labels, bb_labels, pmask_labels = data.load_test_next_batch_sq()
		sem_labels = torch.tensor(sem_labels)
		in_X = torch.tensor(X[:, :, :9]).clone().detach()
		in_X = in_X.to(device, dtype=torch.float32)
		global_features, points_features = backbone(in_X)
		predict_sem_labels = seg_net(global_features, global_features)
		predict_sem_labels = torch.squeeze(predict_sem_labels)
		predict_sem_labels = torch.argmax(predict_sem_labels, dim=1)

		predict_sem_labels = predict_sem_labels.numpy()
		sem_labels = sem_labels.numpy()

		for sem_id in Data_Configs.sem_ids:
			sem_idx = np.where(sem_labels == sem_id)
			sem_label = sem_labels[sem_idx]
			predict_sem_label = predict_sem_labels[sem_idx]
			result = (sem_label == predict_sem_label).astype(int)

			TP_FP_Total[sem_id]['TP'] += np.sum(result)
			TP_FP_Total[sem_id]['FP'] += len(result) - np.sum(result)
			TP_FP_Total[sem_id]['Total'] += len(result)

	pre_all = []
	rec_all = []
	for sem_id, sem_name in zip(Data_Configs.sem_ids, Data_Configs.sem_names):
		TP = TP_FP_Total[sem_id]['TP']
		FP = TP_FP_Total[sem_id]['FP']
		Total = TP_FP_Total[sem_id]['Total']
		pre = float(TP) / (TP + FP + 1e-8)
		rec = float(TP) / (Total + 1e-8)

		if Total > 0:
			pre_all.append(pre)
			rec_all.append(rec)


	print("precision\n", pre_all)
	print("recall\n", rec_all)
	return pre_all, rec_all




def eva_IoU(model_path, data: Data_S3DIS, device):
	backbone = torch.load(model_path["backbone"], map_location=torch.device('cpu'))
	backbone.eval()
	box_net = torch.load(model_path["box_net"], map_location=torch.device('cpu'))
	box_net.eval()
	batch = data.total_test_batch_num
	for i in range(batch):
		X, sem_labels, ins_labels, psem_labels, bb_labels, pmask_labels = data.load_test_next_batch_sq()
		sem_labels = torch.tensor(sem_labels)
		in_X = torch.tensor(X[:, :, :9]).clone().detach()
		in_X = in_X.to(device, dtype=torch.float32)
		global_features, points_features = backbone(in_X)
		predict_boxes, _ = box_net(global_features)
		predict_boxes = predict_boxes.numpy()
		y_bbvert_pred_new, pred_bborder = Ops.box_association(X, bb_labels, predict_boxes, label='use_all_ce_l2_iou')
		avg_ioU = 0.0
		valid_instances = 0
		for j in range(data.ins_max_num):
			pred_box = y_bbvert_pred_new[i, j]		# [2, 3] rectangular
			gt_box = bb_labels[i, j]
			if not np.any(gt_box == 0.):			# some gt boxes are true
				valid_instances += 1
				intersect_leftbottom = np.max(pred_box[0], gt_box[0])
				insersect_rightup = np.min(pred_box[1], gt_box[1])
				if np.any(insersect_rightup > intersect_leftbottom):
					pred_volumn = np.prod(pred_box[1] - pred_box[0])
					gt_volumn = np.prod(gt_box[1] - gt_box[0])
					intersect_volumn = np.prod(insersect_rightup - intersect_leftbottom)
					ioU =intersect_volumn / (pred_volumn + gt_volumn - intersect_volumn)
					avg_ioU += ioU
		avg_ioU /= valid_instances
		print("ioU: {}".format(avg_ioU))
		return avg_ioU









