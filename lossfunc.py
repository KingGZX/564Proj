import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
from helper_s3dis import Data_Configs


class Ops:
    @staticmethod
    # @title gather_tensor_along_2nd_axis_index_select_ver.
    def gather_tensor_along_2nd_axis(pred_pre, indice):
        """
        :param pred_pre:  [Batch, 24, 2, 3],unordered pred bbox
        :param indice:    [Batch, #col_index=24], store col_ind, 从左到右，每一列的行索引
        :target:  重新根据indice对pred_pre的 unordered_bbox 进行对gt box关联
        ;return:  pred_new, [Batch, 24, 2, 3], ordered pred bbox
        """
        batch = pred_pre.shape[0]
        [_, ins_max_num, d1, d2] = pred_pre.shape
        batch_range = torch.arange(0, batch).type(torch.IntTensor)
        batch_range_flat = batch_range.view(-1, 1)
        batch_range_flat_repeat = batch_range_flat.repeat(1, int(ins_max_num))
        batch_range_flat_repeat = batch_range_flat_repeat.view(-1)

        indice_2d_flat = indice.view(-1)
        indice_2d_flat_repeat = batch_range_flat_repeat * int(ins_max_num) + indice_2d_flat
        pred_pre = pred_pre.view(-1, int(d1), int(d2))
        pred_new = pred_pre.index_select(0, indice_2d_flat_repeat)
        pred_new = pred_new.view(batch, -1, int(d1), int(d2))
        return pred_new

    @staticmethod
    def hungarian(loss_matrix, bb_gt):
        box_mask = np.array([[0, 0, 0], [0, 0, 0]])

        def assign_mappings_valid_only(cost, gt_boxes):
            # return ordering : batch_size x num_instances
            loss_total = 0.
            batch_size, num_instances = cost.shape[:2]
            ordering = np.zeros(shape=[batch_size, num_instances]).astype(np.int32)
            for idx in range(batch_size):
                ins_gt_boxes = gt_boxes[idx]
                ins_count = 0
                for box in ins_gt_boxes:
                    if np.array_equal(box, box_mask):
                        break
                    else:
                        ins_count += 1
                valid_cost = cost[idx][:ins_count]
                row_ind, col_ind = linear_sum_assignment(valid_cost)
                unmapped = num_instances - ins_count
                if unmapped > 0:
                    rest = np.array(range(ins_count, num_instances))
                    row_ind = np.concatenate([row_ind, rest])
                    unmapped_ind = np.array(list(set(range(num_instances)) - set(col_ind)))
                    col_ind = np.concatenate([col_ind, unmapped_ind])

                loss_total += cost[idx][row_ind, col_ind].sum()
                ordering[idx] = np.reshape(col_ind, [1, -1])
            return ordering, (loss_total / float(batch_size * num_instances)).astype(np.float32)

        lm = loss_matrix.detach().numpy()
        bgt = bb_gt.detach().numpy()
        ordering, loss_total = assign_mappings_valid_only(lm, bgt)

        return ordering, loss_total

    # @title 一键复制assciation ver.

    # @title C_ed
    @staticmethod
    def box_distance_cost(box_labels, predict_boxes):
        """
        :param box_labels:  ground truth bounding boxes in the shape [Batch, (24)b_num, 2, 3]
        :param predict_boxes:   pred bounding boxes in the shape [Batch, 24, 2, 3]
        :return:  [Batch, b_num, b_num],
                for a matrix [b_num, b_num] is the euclidean distance between pred box i and gt box j
        !! 24 >= b_num !!
        """

        # in association [Batch, 24, 24, 2, 3]
        box_labels = box_labels[:, :, None, :, :].repeat(1, 1, Data_Configs.ins_max_num, 1, 1)  # j
        predict_boxes = predict_boxes[:, None, :, :, :].repeat(1, Data_Configs.ins_max_num, 1, 1, 1)  # i

        """
        if we fix the dimension 0 and 1.
        a vector [6] is coord of each box
        then a [b_num, 6] matrix in predict_boxes is a full group of predicted boxes.
        however, the same matrix in labels is a copy of a specific label box
        in this way, we can directly calculate element wise square and then calculate the mean by row
        """
        # convert both in shape [Batch, b_num, 24, 6]
        box_labels = box_labels.view(-1, Data_Configs.ins_max_num, Data_Configs.ins_max_num, 6)
        predict_boxes = predict_boxes.view(-1, Data_Configs.ins_max_num, Data_Configs.ins_max_num, 6)
        euc_distances = torch.square(predict_boxes - box_labels)
        C_ed = torch.mean(euc_distances, dim=-1)
        return C_ed

    # @title hard_soft_mask
    @staticmethod
    def hard_soft_mask(X, box_labels, predict_boxes):
        """
        :param X:   raw points features, like coordinates rgb kind of things    [Batch, p_num, 9]
        :param box_labels:    [Batch, 24, 2, 3]
        :param predict_boxes:   [Batch, 24, 2, 3]
        :return gt_hard_prob:   float32, [Batch, 24, p_num], q^j which indicates whether a point is inside j-th ground truth box
        :return pred_soft_prob:   float32, [Batch, 24, p_num], q^i which indicates whether a point is inside i-th pred box
        """

        """we want output (prob matrix) has the shape [Batch, 24, p_num] each row i represents the points in the box 
        i"""
        coords = X[:, :, :3]  # [Batch, p_num, 3]
        # we have 24 feature maps, each map is a whole [p_num, 3] points coordinates
        coords_expand = coords[:, None, :, :].repeat(1, Data_Configs.ins_max_num, 1, 1)

        """ 
        hard-binary vector qj (R^N) which indicates whether a point is inside j-th ground truth box 
        if one point is in the ground truth box then (point - box_min) * (box_max - point) is not negative
        only coords on axises xyz of one point are all between min and max, only 3True= all1 in row i(dimension1)
        """
        min_coord = box_labels[:, :, 0, :]  # both are [Batch, 24, 3]
        max_coord = box_labels[:, :, 1, :]
        # we have 24 features maps, each map is the same box min or max coordinate
        min_coords = min_coord[:, :, None, :].repeat(1, 1, coords.shape[1], 1)
        max_coords = max_coord[:, :, None, :].repeat(1, 1, coords.shape[1], 1)
        gt1 = coords_expand - min_coords
        gt2 = max_coords - coords_expand
        gt_hard_prob = gt1 * gt2  # every axis
        # shape is still [Batch, 24, p_num, 3], for a fixed box, its matrix [p_num, 3],
        # if one row i is all positive then we can say the point i is in the specific box
        gt_hard_prob = torch.ge(gt_hard_prob, 0.).type(torch.FloatTensor)
        # shape is still [Batch, 24, p_num, 3], fix dimension 0 and 1, a matrix [p_num, 3] is filled with 0false and 1true
        # if one row i is all 1 then we can determine that i th point is in the specific box(dimension 1)
        gt_hard_prob = torch.eq(torch.mean(gt_hard_prob, dim=-1), 1.).type(
            torch.FloatTensor)  # shape [Batch, b_num, p_num]

        """ 
        sort-binary vector qi (R^N) which indicates whether a point is inside i-th pred box 
        纯按公式
        """
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
        pred_prob = torch.min(prob, dim=-1)[0]  # torch.min -> minimum value and coorespinding index, so [0]

        return gt_hard_prob, pred_prob

    # @title C_sIoU + C_ce
    @staticmethod
    def box_sIOU_CEntrophy_cost(gt_hard_prob, pred_prob):
        """
        :param gt_hard_prob:  [Batch, 24, p_num], q^j
        :return C_sIoU:
        :return C_ce: Cross Entrophy Cost matrix shape [Batch, b_num, b_num]
        """

        # C_sIoU
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

        # C_ce
        C_ce = - gt_hard_prob * torch.log(pred_prob + 1e-6) - (1 - gt_hard_prob) * torch.log(1 - pred_prob + 1e-6)
        C_ce = torch.mean(C_ce, dim=-1)

        return C_sIoU, C_ce

    # @title bbox_association
    @staticmethod
    def box_association(X, box_labels, predict_boxes, label=''):
        """
        :param box_labels:    [Batch, 24, 2, 3]
        :param predict_boxes:   [Batch, 24, 2, 3]
        :param label: definition of bbox assciation cost depends on label
        :target:  calculate the cost matrix used in assciation and pair ground truth box and pred box because points are unordered
        :return y_bbvert_pred_new:
        :return pred_bborder:
        """

        C_ed = Ops.box_distance_cost(box_labels, predict_boxes)

        gt_hard_prob, pred_prob = Ops.hard_soft_mask(X, box_labels, predict_boxes)  # [Batch, 24, p_num]
        # expanding dimensions and repeat can help us vectorize operations of i th predict box and j th label box
        gt_hard_prob = gt_hard_prob[:, :, None, :].repeat(1, 1, Data_Configs.ins_max_num, 1)  # [Batch, 24, 24, p_num]
        pred_prob = pred_prob[:, None, :, :].repeat(1, Data_Configs.ins_max_num, 1, 1)  # [Batch, 24, 24, p_num]

        C_sIoU, C_ce = Ops.box_sIOU_CEntrophy_cost(gt_hard_prob, pred_prob)

        if label == 'use_all_ce_l2_iou':
            C_associate = C_ce + C_ed + C_sIoU
        elif label == 'use_both_ce_l2':
            C_associate = C_ce + C_ed
        elif label == 'use_both_ce_iou':
            C_associate = C_ce + C_sIoU
        elif label == 'use_both_l2_iou':
            C_associate = C_ed + C_sIoU
        elif label == 'use_only_ce':
            C_associate = C_ce
        elif label == 'use_only_l2':
            C_associate = C_ed
        elif label == 'use_only_iou':
            C_associate = C_sIoU
        else:
            C_associate = None
            print('association label error!')
            exit()

        pred_bborder, association_score_min = Ops.hungarian(loss_matrix=C_associate, bb_gt=box_labels)
        pred_bborder = torch.tensor(pred_bborder, dtype=torch.int32)
        y_bbvert_pred_new = Ops.gather_tensor_along_2nd_axis(predict_boxes, pred_bborder)
        return y_bbvert_pred_new, pred_bborder

    @staticmethod
    def bbscore_association(y_bbscore_pred_raw, pred_bborder):
        y_bbscore_pred_raw = y_bbscore_pred_raw.unsqueeze(-1).unsqueeze(-1)
        y_bbscore_pred_new = Ops.gather_tensor_along_2nd_axis(y_bbscore_pred_raw, pred_bborder)

        y_bbscore_pred_new = y_bbscore_pred_new.view(-1, y_bbscore_pred_new.shape[1])
        return y_bbscore_pred_new

    @staticmethod
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
        psem_labels = torch.argmax(psem_labels, dim=1)
        loss = loss_fn(predict_point_mask, psem_labels)
        return loss

    # @title bboxloss

    #  bbox loss
    @staticmethod
    def box_loss(X, box_labels, predict_boxes, label=""):
        """
        :param label: loss seting
        :param predict_boxes:
        :param box_labels: [Batch, 24, 2, 3]
        :target: calculte cost for valid, indicate the effeiciency of pred box, after prediction, only valid
        :return bbox_loss:
        """

        box_labels_pos = box_labels.view(-1, Data_Configs.ins_max_num,
                                         6)  # [Batch, 24, 6], dimensiondimension 1 is one box
        box_labels_pos = torch.ge(torch.sum(box_labels_pos, dim=-1), 0.).type(torch.FloatTensor)
        # [Batch, 24], dimension 0 filled with 0false and 1true which means whether the box in gt box is valid
        T = torch.sum(box_labels_pos)  # int  how many the ture and valid ground Truth boxes are

        """ hard_soft_mask """
        # attetion: the dimensions of gt_hard_prob and pred_prob
        gt_hard_prob, pred_prob = Ops.hard_soft_mask(X, box_labels, predict_boxes)  # [Batch, 24, p_num]

        """ l_ed_valid """
        box_labels = box_labels.view(-1, Data_Configs.ins_max_num, 6)
        predict_boxes_all = predict_boxes.view(-1, Data_Configs.ins_max_num, 6)
        euc_distances = torch.square(predict_boxes_all - box_labels)
        l_ed_all = torch.mean(euc_distances, dim=-1)  # [Batch 24]
        l_ed_pos = torch.sum(l_ed_all * box_labels_pos) / T
        # # to minimize the 3D volumn of invalid/negative bboxes, it serves as a regularizer to penalize false pred
        # bboxes # it turns out to be quite helpful, but not discussed in the paper
        box_labels_neg = (1. - box_labels_pos)
        predict_boxes_neg = box_labels_neg[:, :, None, None].repeat(1, 1, 2, 3) * predict_boxes  # [Batch, 24, 2, 3]
        predict_boxes_neg = (predict_boxes_neg[:, :, 0, :] - predict_boxes_neg[:, :, 1, :]) ** 2
        l_ed_neg = torch.sum(predict_boxes_neg) / (torch.sum(box_labels_neg) + 1e-6)
        # sum of pos and neg
        l_ed_valid = l_ed_pos + l_ed_neg

        """ l_sIoU_valid  """
        numerator = torch.sum(gt_hard_prob * pred_prob, dim=-1)
        denominator = torch.sum(gt_hard_prob, dim=-1) + torch.sum(pred_prob, dim=-1) - numerator + 1e-6
        l_sIoU_all = -1.0 * (numerator / denominator)
        l_sIoU_valid = torch.sum(l_sIoU_all * box_labels_pos) / T

        """ l_CE_valid  """
        box_labels_pos_tp = box_labels_pos[:, :, None].repeat(1, 1, X.shape[1])  # [Batch, 24, p_num]
        l_ce_all = - gt_hard_prob * torch.log(pred_prob + 1e-6) - (1. - gt_hard_prob) * torch.log(1. - pred_prob + 1e-6)
        l_ce_valid = torch.sum(l_ce_all * box_labels_pos_tp) / torch.sum(box_labels_pos_tp)

        if label == 'use_all_ce_l2_iou':
            bbox_loss = l_ce_valid + l_ed_valid + l_sIoU_valid
        elif label == 'use_both_ce_l2':
            bbox_loss = l_ce_valid + l_ed_valid
        elif label == 'use_both_ce_iou':
            bbox_loss = l_ce_valid + l_sIoU_valid
        elif label == 'use_both_l2_iou':
            bbox_loss = l_ed_valid + l_sIoU_valid
        elif label == 'use_only_ce':
            bbox_loss = l_ce_valid
        elif label == 'use_only_l2':
            bbox_loss = l_ed_valid
        elif label == 'use_only_iou':
            bbox_loss = l_sIoU_valid
        else:
            bbox_loss = None
            print('bbox loss label error!')
            exit()
        return bbox_loss  # + l_ed_valid, l_sIoU_valid, l_ce_valid

    @staticmethod
    def get_loss_bbscore(y_bbscore_pred, box_labels):
        bb_num = int(box_labels.shape[1])

        # helper -> the valid bbox
        box_labels_pos = box_labels.view(-1, Data_Configs.ins_max_num,
                                         6)  # [Batch, 24, 6], dimensiondimension 1 is one box
        box_labels_pos = torch.ge(torch.sum(box_labels_pos, dim=-1), 0.).type(torch.FloatTensor)

        # bbox score loss
        bbox_loss_score = torch.mean(-box_labels_pos * torch.log(y_bbscore_pred + 1e-8)
                                     - (1. - box_labels_pos) * torch.log(1. - y_bbscore_pred + 1e-8))
        return bbox_loss_score

    @staticmethod
    def get_loss_pmask(X_pc, y_pmask_pred, Y_pmask):
        points_num = X_pc.shape[1]

        Y_pmask_helper = Y_pmask.sum(dim=-1)
        Y_pmask_helper = (Y_pmask_helper > 0).float()
        Y_pmask_helper = Y_pmask_helper.unsqueeze(dim=-1).repeat(1, 1, points_num)
        Y_pmask = Y_pmask * Y_pmask_helper
        y_pmask_pred = y_pmask_pred * Y_pmask_helper

        # focal loss
        alpha = 0.75
        gamma = 2
        pmask_loss_focal_all = -Y_pmask * alpha * ((1. - y_pmask_pred) ** gamma) * torch.log(y_pmask_pred + 1e-8) - (
                1. - Y_pmask) * (1. - alpha) * (y_pmask_pred ** gamma) * torch.log(1. - y_pmask_pred + 1e-8)
        pmask_loss_focal = torch.sum(pmask_loss_focal_all * Y_pmask_helper) / torch.sum(Y_pmask_helper)

        pmask_loss = 30 * pmask_loss_focal

        return pmask_loss
