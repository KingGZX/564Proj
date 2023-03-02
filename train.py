import torch.optim as optim
import torch.nn as nn
import torch
from helper_s3dis import Data_S3DIS
import os
from lossfunc import Ops


def train(model: list, data: Data_S3DIS):
    # setting hyperparameters
    backbone, seg_net, box_net, mask_net = model
    start_epoch = 0
    optimizer = optim.Adam([{"params": backbone.parameters()}, {"params": seg_net.parameters()},
                            {"params": box_net.parameters()}, {"params": mask_net.parameters()}],
                           lr=5e-4)

    checkpoint_path = "./checkpoint/***.pkl"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        backbone.load_state_dict(checkpoint['backbone_state_dict'])  # load the latest training parameters
        seg_net.load_state_dict(checkpoint['seg_net_state_dict'])
        box_net.load_state_dict(checkpoint['box_net_state_dict'])
        mask_net.load_state_dict(checkpoint['mask_net_state_dict'])
        start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch, start_epoch + 50):
        data.shuffle_train_files(epoch)
        batch = data.total_train_batch_num
        for i in range(batch):
            X, sem_labels, ins_labels, psem_labels, bb_labels, pmask_labels = data.load_train_next_batch()
            X = torch.tensor(X)
            sem_labels = torch.tensor(sem_labels)
            ins_labels = torch.tensor(ins_labels)
            psem_labels = torch.tensor(psem_labels)
            bb_labels = torch.tensor(bb_labels)
            pmask_labels = torch.tensor(pmask_labels)
            global_features, points_features = backbone(torch.tensor(X[:, :, :9]))
            predict_sem_labels = seg_net(global_features, points_features)
            predict_boxes, predict_boxes_scores = box_net(global_features)
            predict_point_mask = mask_net(global_features, points_features, predict_boxes, predict_boxes_scores)
            loss1 = Ops.semantic_loss(psem_labels, predict_sem_labels)
            y_bbvert_pred_new, pred_bborder = Ops.box_association(X, bb_labels, predict_boxes, label='use_all_ce_l2_iou')
            y_bbscore_pred_new = Ops.bbscore_association(predict_boxes_scores, pred_bborder)
            loss2 = Ops.box_loss(X, bb_labels, y_bbvert_pred_new, label='use_all_ce_l2_iou')
            loss3 = Ops.get_loss_bbscore(y_bbscore_pred_new, bb_labels)
            loss4 = Ops.get_loss_pmask(X, predict_point_mask, pmask_labels)
            loss = loss1 + loss2 + loss3 + loss4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())


def eval():
    pass
