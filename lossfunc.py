import torch
import torch.nn as nn
from helper_s3dis import Data_Configs
import torch.nn.functional as F


def semantic_loss(psem_labels, predict_point_mask):
    """
    :param psem_labels:  shape [B, p_num, semantic_num] which indicates whether a point belongs to a specific semantic
    :param predict_point_mask: network output
    :return:    loss
    """
    loss_fn = nn.CrossEntropyLoss()
    """
    for torch CrossEntropy, we want the shape of the input [Batch, Classes, d1, ..., dk]
    """
    psem_labels = torch.permute(psem_labels, [0, 2, 1])
    predict_point_mask = torch.permute(predict_point_mask, [0, 2, 1])
    loss = loss_fn(psem_labels, predict_point_mask)
    return loss


def box_distance_cost(box_labels, predict_boxes):
    """
    :param box_labels:  bounding boxes in the shape [Batch, b_num, 2, 3]
    :param predict_boxes:
    :return:  [Batch, b_num, b_num], for a matrix [b_num, b_num] is the euclidean distance between predicted box i and
    label j
    """

    box_labels = box_labels[:, :, None, :, :].repeat(1, 1, Data_Configs.ins_max_num, 1, 1)
    predict_boxes = predict_boxes[:, None, :, :, :].repeat(1, Data_Configs.ins_max_num, 1, 1, 1)
    # convert both in shape [Batch, b_num, b_num, 6]
    """
    if we fix the dimension 0 and 1.
    then a [b_num, 6] matrix in predict_boxes is a full group of predicted boxes.
    however, the same matrix in labels is a copy of a specific label box
    in this way, we can directly calculate element wise square and then calculate the mean by row
    """
    box_labels = box_labels.view(-1, Data_Configs.ins_max_num, Data_Configs.ins_max_num, 6)
    predict_boxes = predict_boxes.view(-1, Data_Configs.ins_max_num, Data_Configs.ins_max_num, 6)
    euc_distances = torch.square(predict_boxes - box_labels)
    C_ed = torch.mean(euc_distances, dim=-1)
    return C_ed


def box_score_loss():

    pass
