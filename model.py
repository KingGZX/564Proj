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

        predict_sem_labels = F.softmax(torch.permute(torch.squeeze(f6), [0, 2, 1]))
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
        # output shape is (B, H, 2, 3), but we don't know whether this box is represented correctly

        min_coord = torch.min(boxes, dim=2)[0][:, :, None, :]  # shape is [B, N, 1, 3]
        max_coord = torch.max(boxes, dim=2)[0][:, :, None, :]
        predict_boxes = torch.concatenate([min_coord, max_coord], dim=2)

        """
        branch2: predict box scores
        """
        f4 = self.relu5(self.fc5(f2))
        predict_box_scores = F.sigmoid(self.fc6(f4))

        return predict_boxes, predict_box_scores


class Pmask_Net(nn.Module):
    def __init__(self):
        super(Pmask_Net, self).__init__()

    def forward(self, global_features, points_features, predict_boxes, predict_box_scores):
        pass
