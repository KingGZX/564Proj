import numpy as np
import scipy.stats
import os
import scipy.io
import torch
import glob
import h5py
from utils import *
from helper_s3dis import Data_Configs, Data_S3DIS
from Transformer import *


def eva_accu(backbone, seg_net, data: Data_S3DIS, device):
    """
	accuracy of semantic: backbone + seg_net

	"""
    backbone.eval()
    seg_net.eval()
    batch = data.total_test_batch_num

    TP_FP_Total = {}
    for sem_id in Data_Configs.sem_ids:
        TP_FP_Total[sem_id] = {}
        TP_FP_Total[sem_id]['TP'] = 0
        TP_FP_Total[sem_id]['FP'] = 0
        TP_FP_Total[sem_id]['Total'] = 0

    for i in range(batch):
        X, sem_labels, ins_labels, psem_labels, bb_labels, pmask_labels = data.load_test_next_batch_sq()  # , psem_labels, bb_labels, pmask_labels#
        sem_labels = torch.tensor(sem_labels)
        in_X = torch.tensor(X[:, :, :9]).clone().detach()
        in_X = in_X.to(device, dtype=torch.float32)
        global_features, points_features = backbone(in_X)

        predict_sem_labels = seg_net(global_features, points_features)
        predict_sem_labels = torch.squeeze(predict_sem_labels)
        predict_sem_labels = torch.argmax(predict_sem_labels, dim=-1)

        predict_sem_labels = predict_sem_labels.cpu().numpy()
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
    for sem_id, sem_name in zip(Data_Configs.sem_ids, Data_Configs.sem_names):
        TP = TP_FP_Total[sem_id]['TP']
        FP = TP_FP_Total[sem_id]['FP']
        Total = TP_FP_Total[sem_id]['Total']
        pre = float(TP) / (TP + FP + 1e-8)

        if Total > 0:
            pre_all.append(pre)
    print("precision    ", pre_all)
    return pre_all


def eva_accu_transformer(transformer, seg_net, data: Data_S3DIS, device):
    # transformer = transformer.cpu()
    # seg_net = seg_net.cpu()
    transformer.eval()
    seg_net.eval()
    batch = data.total_test_batch_num

    TP_FP_Total = {}
    for sem_id in Data_Configs.sem_ids:
        TP_FP_Total[sem_id] = {}
        TP_FP_Total[sem_id]['TP'] = 0
        TP_FP_Total[sem_id]['FP'] = 0
        TP_FP_Total[sem_id]['Total'] = 0

    for i in range(batch):
        X, sem_labels, ins_labels, psem_labels, bb_labels, pmask_labels = data.load_test_next_batch_sq()  # , psem_labels, bb_labels, pmask_labels#

        idx, points = farthest_point_sampling(X)
        points = points.to(device, dtype=dtype)
        X = torch.tensor(X)
        sem_labels = torch.tensor(sem_labels)
        X = X.to(device, dtype=dtype)
        global_features, points_features = transformer(X, points, idx)  # "cpu"
        predict_sem_labels = seg_net(global_features, points_features)

        predict_sem_labels = torch.squeeze(predict_sem_labels)
        predict_sem_labels = torch.argmax(predict_sem_labels, dim=-1)

        predict_sem_labels = predict_sem_labels.cpu().numpy()
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
    for sem_id, sem_name in zip(Data_Configs.sem_ids, Data_Configs.sem_names):
        TP = TP_FP_Total[sem_id]['TP']
        FP = TP_FP_Total[sem_id]['FP']
        Total = TP_FP_Total[sem_id]['Total']
        pre = float(TP) / (TP + FP + 1e-8)

        if Total > 0:
            pre_all.append(pre)
    print("precision    ", pre_all)
    return pre_all


def eva_IoU(backbone, box_net, data: Data_S3DIS, device):
    backbone.eval()
    box_net.eval()
    batch = data.total_test_batch_num
    avg_ioU = 0.0
    valid_instances = 0
    for i in range(batch):
        X, sem_labels, ins_labels, psem_labels, bb_labels, pmask_labels = data.load_test_next_batch_sq()
        sem_labels = torch.tensor(sem_labels)
        in_X = torch.tensor(X[:, :, :9]).clone().detach()
        in_X = in_X.to(device, dtype=torch.float32)
        global_features, points_features = backbone(in_X)
        predict_boxes, _ = box_net(global_features)
        # predict_boxes = predict_boxes.cpu().numpy()
        X = torch.tensor(X)
        bb_labels = torch.tensor(bb_labels)
        X = X.to(device)
        bb_labels = bb_labels.to(device)
        y_bbvert_pred_new, pred_bborder = Ops.box_association(X, bb_labels, predict_boxes, label='use_all_ce_l2_iou')
        y_bbvert_pred_new = y_bbvert_pred_new.cpu().detach().numpy()
        bb_labels = bb_labels.cpu().numpy()
        for k in range(data.test_batch_size):
            for j in range(data.ins_max_num):
                pred_box = y_bbvert_pred_new[k, j]  # [2, 3] rectangular
                gt_box = bb_labels[k, j]
                if not np.any(gt_box == 0):  # some gt boxes are true
                    valid_instances += 1
                    intersect_leftbottom = np.max((pred_box[0], gt_box[0]), axis=0)
                    insersect_rightup = np.min((pred_box[1], gt_box[1]), axis=0)
                    if np.all(insersect_rightup > intersect_leftbottom):
                        pred_volumn = np.prod(pred_box[1] - pred_box[0])
                        gt_volumn = np.prod(gt_box[1] - gt_box[0])
                        intersect_volumn = np.prod(insersect_rightup - intersect_leftbottom)
                        ioU = intersect_volumn / (pred_volumn + gt_volumn - intersect_volumn)
                        avg_ioU += ioU
    avg_ioU /= valid_instances
    print("ioU: {}".format(avg_ioU))
    return avg_ioU
