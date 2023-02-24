import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_s3dis import Data_Configs


class Backbone_BoNet(nn.Module):
    def __init__(self, points_feature=Data_Configs.points_cc):
        super(Backbone_BoNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, points_feature))
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.relu3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.relu4 = nn.LeakyReLU()
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=1)
        self.relu5 = nn.LeakyReLU()

    def forward(self, points):
        """
        param:
            points: [B, N, dimension] , batch size, points number, feature of a point (like coordinate, rgb etc.)
        """
        points = torch.unsqueeze(points, dim=1)  # then points become [B, 1, N, dimension]
        f1 = self.relu1(self.conv1(points))
        f2 = self.relu2(self.conv2(f1))
        f3 = self.relu3(self.conv3(f2))
        f4 = self.relu4(self.conv4(f3))
        f5 = self.relu5(self.conv5(f4))  # output should have the shape [B, 1024, N, 1]
        """
        get the global features in the shape (B, 1024) and points features (B, N, 1024)
        """
        f5 = torch.permute(f5, [0, 2, 3, 1])  # in the shape [B, N, 1, 1024]
        global_features = torch.max(f5, dim=1)[0].squeeze()  # torch.max return tuple (values, indices)
        points_features = torch.squeeze(f5)
        # encoded features returned
        return global_features, points_features


class Semantic_Net(nn.Module):
    """
    predict the semantic label of each point
    """

    def __init__(self, input_features=1024, output_features=128, dropout=0.5):
        super(Semantic_Net, self).__init__()
        self.fc1 = nn.Linear(input_features, 256)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(256, output_features)
        self.relu2 = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(input_features + output_features, 512, kernel_size=1)
        self.relu3 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.relu4 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(256, 128, kernel_size=1)
        self.relu5 = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv4 = nn.Conv2d(128, Data_Configs.sem_num, kernel_size=1)

    def forward(self, global_features, points_features):
        f1 = self.relu1(self.fc1(global_features))
        f2 = self.relu2(self.fc2(f1))
        # after transformation, global features become [Batch, 128]

        f2 = f2[:, None, None, :]
        f2 = f2.repeat(1, points_features.shape[1], 1, 1)
        points_features = torch.unsqueeze(points_features, dim=2)
        all_together_features = torch.concatenate([points_features, f2], dim=3)  # [Batch, points_num, 1, 1024 + 128]
        all_together_features = torch.permute(all_together_features, [0, 3, 1, 2])

        f3 = self.relu3(self.conv1(all_together_features))
        f4 = self.relu4(self.conv2(f3))
        f5 = self.dropout(self.relu5(self.conv3(f4)))
        f6 = self.conv4(f5)  # [Batch, semantic_num, points, 1]

        # no need to use softmax here since CrossEntropy Loss can automatically do this for us
        # predict_sem_labels = F.softmax(torch.permute(torch.squeeze(f6), [0, 2, 1]), dim=2)  # .. CHECK
        predict_sem_labels = torch.permute(torch.squeeze(f6), [0, 2, 1])
        return predict_sem_labels


class BBox_Net(nn.Module):
    def __init__(self, in_features=1024, max_boxes=Data_Configs.ins_max_num):
        super(BBox_Net, self).__init__()
        self.mbox = max_boxes

        self.fc1 = nn.Linear(in_features, 512)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.LeakyReLU()

        # branch one to predict box positions
        self.fc3 = nn.Linear(256, 256)
        self.relu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(256, self.mbox * 2 * 3)

        # branch two to predict box scores
        self.fc5 = nn.Linear(256, 256)
        self.relu5 = nn.LeakyReLU()
        self.fc6 = nn.Linear(256, self.mbox)

    def forward(self, global_features):
        """
        :param global_features: in the shape [B, 1024]
        :return: predicted bounding box coordinates and corresponding box scores which reflect box quality
        """
        f1 = self.relu1(self.fc1(global_features))
        f2 = self.relu2(self.fc2(f1))

        """
        branch1: predict branches
        """
        f3 = self.relu3(self.fc3(f2))
        boxes = self.fc4(f3)
        boxes = boxes.view(boxes.shape[0], -1, 2, 3)
        print(boxes.shape)
        # output shape is (B, H, 2, 3), but we don't know whether this box is represented correctly

        min_coord = torch.min(boxes, dim=2)[0][:, :, None, :]  # shape is [B, N, 1, 3]
        max_coord = torch.max(boxes, dim=2)[0][:, :, None, :]
        predict_boxes = torch.concatenate([min_coord, max_coord], dim=2)

        """
        branch2: predict box scores
        """
        f4 = self.relu5(self.fc5(f2))
        predict_box_scores = torch.sigmoid(self.fc6(f4))

        return predict_boxes, predict_box_scores


class Pmask_Net(nn.Module):
    def __init__(self, input_features=1024, output_features=128):
        super(Pmask_Net, self).__init__()
        self.mbox = Data_Configs.ins_max_num
        self.fc1 = nn.Linear(input_features, 256)
        self.relu1 = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, input_features))
        self.relu2 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(512, 128, kernel_size=1)
        self.relu3 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(128, output_features, kernel_size=1)
        self.relu4 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(1, 64, kernel_size=(1, output_features + 7))     # 7 is box coordinates(6) + box score(1)
        self.relu5 = nn.LeakyReLU()
        self.conv5 = nn.Conv2d(64, 32, kernel_size=1)
        self.relu6 = nn.LeakyReLU()
        self.conv6 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, global_features, points_features, predict_boxes, predict_box_scores):
        """
        :param global_features:  shape [Batch, 1024]
        :param points_features:  shape [Batch, points_num, 1024]
        :param predict_boxes:    shape [Batch, boxes_num, 2, 3]
        :param predict_box_scores:  shape  [Batch, boxes_num]
        :return:    shape [Batch, boxes_num, points_num]  which is same as pmask_labels in load_batch in train.py
        """
        batch, p_num, p_feature_num = points_features.shape

        # change the global and points features and concat them
        g1 = self.relu1(self.fc1(global_features))[:, :, None, None]   # shape [Batch, 256, 1, 1]
        g1 = g1.repeat(1, 1, p_num, 1)  # shape [Batch, 256, p_num, 1]
        g2 = self.relu2(self.conv1(points_features[:, None, :, :]))    # shape [Batch, 256, p_num, 1]
        features = torch.concatenate([g2, g1], dim=1)                 # shape [Batch, 512, p_num, 1]
        # print(features.shape)
        f1 = self.relu3(self.conv2(features))
        f2 = self.relu4(self.conv3(f1))         # shape [Batch. 128, p_num, 1]
        new_features = torch.squeeze(f2)        # shape [Batch, 128, p_num]

        boxes = predict_boxes.view(-1, self.mbox, 6)       # reshape [2, 3] rectangular notation to long vector
        # print(boxes.shape)                               # shape [Batch, boxes_num, 6]
        boxes_scores = predict_box_scores[:, :, None]      # shape [Batch, boxes_num, 1]
        box_features = torch.concatenate([boxes, boxes_scores], dim=-1)         # shape [Batch, boxes_num, 7]

        new_features = new_features[:, :, None, :].repeat(1, 1, self.mbox, 1)   # shape [Batch, 128, boxes_num, p_num]
        box_features = torch.permute(box_features, [0, 2, 1])[:, :, :, None]   # shape is [Batch, 7, boxes_num, 1]
        box_features = box_features.repeat(1, 1, 1, p_num)                     # shape is [Batch, 7, boxes_num, p_num]
        mask_features = torch.concatenate([new_features, box_features], dim=1)

        # print(mask_features.shape)  # shape [Batch, 128 + 7, boxes_num, p_num]
        mask_features = mask_features.view(-1, 1, p_num, mask_features.shape[1])  # shape [Batch * boxes_num, 1, p_num,]

        m1 = self.relu5(self.conv4(mask_features))
        m2 = self.relu6(self.conv5(m1))
        predict_point_mask = self.conv6(m2)             # shape is [Batch * boxes_num, 1, p_num, 1]

        predict_point_mask = predict_point_mask.view(-1, self.mbox, p_num)      # shape is [Batch, boxes_num, p_num]
        predict_point_mask = torch.sigmoid(predict_point_mask)
        return predict_point_mask

