import torch
import torch.nn as nn
from helper_s3dis import Data_Configs


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


def box_s_iou_cost(X, box_labels, predict_boxes):
    """
    :param X:   raw points features, like coordinates rgb kind of things    [Batch, p_num, 9]
    :param box_labels:  same as above function, shape [Batch, b_num, 2, 3]
    :param predict_boxes:   same as above
    :return:   Cost matrix shape [Batch, b_num, b_num]
    """
    coords = X[:, :, :3]  # [Batch, p_num, 3]
    """
    we want output (prob matrix) has the shape [Batch, b_num, p_num]
        each row i represents the points in the box i
    """
    # Firstly, calculate hard-binary vector qj (R^N) which indicates whether a point is inside j th true box
    min_coord = box_labels[:, :, 0, :]  # both are in shape [Batch, b_num, 3]
    max_coord = box_labels[:, :, 1, :]
    # we have b_num feature maps, each map is a whole [p_num, 3] points coordinates
    coords_expand = coords[:, None, :, :].repeat(1, Data_Configs.ins_max_num, 1, 1)
    # we have b_num features maps, each map is the same box min or max coordinate
    min_coords = min_coord[:, :, None, :].repeat(1, 1, coords.shape[1], 1)
    max_coords = max_coord[:, :, None, :].repeat(1, 1, coords.shape[1], 1)
    # we know that if one point is in the ground truth box then (point - box_min) * (box_max - point) is positive
    gt1 = coords_expand - min_coords
    gt2 = max_coords - coords_expand
    gt_hard_prob = gt1 * gt2
    # shape is still [Batch, b_num, p_num, 3], for a fixed box, its matrix [p_num, 3],
    # if one row i is all positive then we can say the point i is in the specific box
    gt_hard_prob = torch.ge(gt_hard_prob, 0.).type(torch.FloatTensor)
    """
    after all operations, gt_hard_prob is in shape [Batch, b_num, p_num, 3]
    fix dimension 0 and 1, a matrix [p_num, 3] is filled with 0 and 1
    and we should know, if one row i is all 1 then we can determine that i th point is in the specific box(dimension 1)
    """
    gt_hard_prob = torch.eq(torch.mean(gt_hard_prob, dim=-1), 1.).type(torch.FloatTensor)  # shape [Batch, b_num, p_num]

    # Secondly, we calculate soft prob cost
    min_coord_pred = predict_boxes[:, :, 0, :]
    max_coord_pred = predict_boxes[:, :, 1, :]
    min_coords_pred = min_coord_pred[:, :, None, :].repeat(1, 1, coords.shape[1], 1)
    max_coords_pred = max_coord_pred[:, :, None, :].repeat(1, 1, coords.shape[1], 1)
    pr1 = min_coords_pred - coords_expand
    pr2 = coords_expand - max_coords_pred
    theta1, theta2 = 100, torch.tensor(20)
    pred = torch.maximum(torch.minimum(theta1 * pr1 * pr2, theta2), -theta2)  # in shape [Batch, b_num, p_num, 3]
    sigmoid = nn.Sigmoid()
    prob = sigmoid(pred)
    pred_prob = torch.min(prob, dim=-1)[0]

    # finally, we can calculate the Cost_sIoU

    """
    expanding dimensions and repeat can help us vectorize operations of i th predict box and j th label box
    """
    gt_hard_prob = gt_hard_prob[:, :, None, :].repeat(1, 1, Data_Configs.ins_max_num, 1)
    pred_prob = pred_prob[:, None, :, :].repeat(1, Data_Configs.ins_max_num, 1, 1)
    numerator = torch.sum(gt_hard_prob * pred_prob, dim=-1)
    """
    An explanation of denominator:
        1. shape is [Batch, b_num, b_num]
        2. notice the repeat operation. For gt_prob, each feature map (all rows) after expanding is about the same box 
        and its corresponding points. while pred_pron, each feature map is all the boxes and corresponding points.
        3. therefore, we use element wise multiplication to achieve qi times qj bar.
    """
    denominator = torch.sum(gt_hard_prob, dim=-1) + torch.sum(pred_prob, dim=-1) - numerator + 1e-6
    C_sIoU = -1.0 * (numerator / denominator)
    return C_sIoU


def box_score_loss():
    pass
