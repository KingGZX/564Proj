import numpy as np
from Transformer import *
from helper_s3dis import Data_S3DIS
import glob
from model import *
import torch
import h5py
from utils import *
from dataplot import Plot
import copy

"""
test the procedures step by step
"""

"""
a = 'Area_1'
con = glob.glob("./data/" + a + '*.h5')
"""


data = Data_S3DIS("./data/", ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6'], ['Area_5'])

from train import train_transformer

net = Transformer(9, 4, 64)
net2 = Semantic_Net()
train_transformer([net, net2], data)



"""
data = Data_S3DIS("./data/Data_S3DIS/", ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6'], ['Area_5'])

from train import train
net = Backbone_BoNet(9)
net2 = Semantic_Net()
net3 = BBox_Net()
net4 = Pmask_Net()
train([net, net2, net3, net4], data)
"""

"""
X, sem_labels, ins_labels, psem_labels, bb_labels, pmask_labels = data.load_train_next_batch()
print(X.shape)

net = Backbone_BoNet(9)

gf, pf = net(torch.tensor(X[:, :, :9]))

print(gf.shape, pf.shape)

net2 = Semantic_Net()
pred_sem_labels = net2(gf, pf)
print(pred_sem_labels.shape)

net3 = BBox_Net()
box, box_s = net3(gf)
print(box.shape, box_s.shape)

net4 = Pmask_Net()
masks = net4(gf, pf, box, box_s)
print(masks.shape)

from lossfunc import *
C_ed = box_distance_cost(torch.tensor(bb_labels), box)
print(C_ed.shape)

C_sIoU = box_s_iou_cost(torch.tensor(X), torch.tensor(bb_labels), box)
print(C_sIoU.shape)
"""

"""
fin = h5py.File("./data/Area_5_hallway_1.h5", 'r')

## (55, 4096, 3)  &    [55, 4096, 9]
coords = fin['coords'][:]
points = fin['points'][:]

batch, num, features1 = coords.shape
_, _, features2 = points.shape

temp = np.zeros((batch, num, features1 + features2))

for i in range(batch):
    pc_xyzrgb = np.concatenate([coords[i], points[i][:, 3:9]], axis=-1)
    min_x = np.min(pc_xyzrgb[:, 0])
    max_x = np.max(pc_xyzrgb[:, 0])
    min_y = np.min(pc_xyzrgb[:, 1])
    max_y = np.max(pc_xyzrgb[:, 1])
    min_z = np.min(pc_xyzrgb[:, 2])
    max_z = np.max(pc_xyzrgb[:, 2])

    ori_xyz = copy.deepcopy(pc_xyzrgb[:, 0:3])  # reserved for final visualization
    pc_xyzrgb[:, 0:1] = (pc_xyzrgb[:, 0:1] - min_x) / np.maximum((max_x - min_x), 1e-3)
    pc_xyzrgb[:, 1:2] = (pc_xyzrgb[:, 1:2] - min_y) / np.maximum((max_y - min_y), 1e-3)
    pc_xyzrgb[:, 2:3] = (pc_xyzrgb[:, 2:3] - min_z) / np.maximum((max_z - min_z), 1e-3)
    pc_xyzrgb = np.concatenate([pc_xyzrgb, ori_xyz], axis=-1)   # [points, 12]
    temp[i] = pc_xyzrgb

X = temp.reshape((-1, features1 + features2))
# print(pc_xyzrgb.shape)


backbone = torch.load("J:\\COMProfile\\Downloads\\pk\\42\\backbone_42.pth", map_location=torch.device('cpu'))
backbone.eval()
seg_net = torch.load("J:\\COMProfile\\Downloads\\pk\\42\\seg_net_42.pth", map_location=torch.device('cpu'))
seg_net.eval()
in_X = torch.tensor(X[None, :, :9], dtype=torch.float32).clone().detach()
f1, f2 = backbone(in_X)
f1 = f1[None, :]
# f2 = f2[None, :, :]
predict_sem_labels = seg_net(f1, f2)
# print(predict_sem_labels.shape)
predict_sem_labels = torch.squeeze(predict_sem_labels)
predict_sem_labels = torch.argmax(predict_sem_labels, dim=1)
predict_sem_labels = predict_sem_labels.numpy()
print(predict_sem_labels.shape)

Plot.draw_pc_semins(fin['coords'][:].reshape(-1, 3), predict_sem_labels, 13)
"""


"""
# test_transformer
fin = h5py.File("./data/Area_5_WC_1.h5", 'r')

# [55, 4096, 3]   &    [55, 4096, 9]
coords = fin['coords'][:]
points = fin['points'][:]

batch, num, features1 = coords.shape

_, _, features2 = points.shape

temp = np.zeros((batch, num, features1 + features2), dtype=np.float32)

for i in range(batch):
    pc_xyzrgb = np.concatenate([coords[i], points[i][:, 3:9]], axis=-1)
    min_x = np.min(pc_xyzrgb[:, 0])
    max_x = np.max(pc_xyzrgb[:, 0])
    min_y = np.min(pc_xyzrgb[:, 1])
    max_y = np.max(pc_xyzrgb[:, 1])
    min_z = np.min(pc_xyzrgb[:, 2])
    max_z = np.max(pc_xyzrgb[:, 2])

    ori_xyz = copy.deepcopy(pc_xyzrgb[:, 0:3])  # reserved for final visualization
    pc_xyzrgb[:, 0:1] = (pc_xyzrgb[:, 0:1] - min_x) / np.maximum((max_x - min_x), 1e-3)
    pc_xyzrgb[:, 1:2] = (pc_xyzrgb[:, 1:2] - min_y) / np.maximum((max_y - min_y), 1e-3)
    pc_xyzrgb[:, 2:3] = (pc_xyzrgb[:, 2:3] - min_z) / np.maximum((max_z - min_z), 1e-3)
    pc_xyzrgb = np.concatenate([pc_xyzrgb, ori_xyz], axis=-1)   # [points, 12]
    temp[i] = pc_xyzrgb

idx, points = farthest_point_sampling(temp)
X = torch.tensor(temp, dtype=torch.float32)

transformer = torch.load("J:\\COMProfile\\Downloads\\pk\\transformer_0411\\transformer_19.pth", map_location=torch.device('cpu'))
transformer.eval()
seg_net = torch.load("J:\\COMProfile\\Downloads\\pk\\transformer_0411\\seg_net_19.pth", map_location=torch.device('cpu'))
seg_net.eval()
global_features, points_features = transformer(X, points, idx, "cpu")
predict_sem_labels = seg_net(global_features, points_features)
predict_sem_labels = torch.squeeze(predict_sem_labels)
predict_sem_labels = torch.argmax(predict_sem_labels, dim=-1)   #[batch, points]
predict_sem_labels = predict_sem_labels.view(-1)
predict_sem_labels = predict_sem_labels.numpy()


Plot.draw_pc_semins(fin['coords'][:].reshape(-1, 3), predict_sem_labels, 13)
"""


"""
# test_transformer: large blocks
fin = h5py.File("./data/Area_5_hallway_1.h5", 'r')

# [55, 4096, 3]   &    [55, 4096, 9]
coords = fin['coords'][:]
points = fin['points'][:]

batch, num, features1 = coords.shape

_, _, features2 = points.shape

temp = np.zeros((batch, num, features1 + features2), dtype=np.float32)

for i in range(batch):
    pc_xyzrgb = np.concatenate([coords[i], points[i][:, 3:9]], axis=-1)
    min_x = np.min(pc_xyzrgb[:, 0])
    max_x = np.max(pc_xyzrgb[:, 0])
    min_y = np.min(pc_xyzrgb[:, 1])
    max_y = np.max(pc_xyzrgb[:, 1])
    min_z = np.min(pc_xyzrgb[:, 2])
    max_z = np.max(pc_xyzrgb[:, 2])

    ori_xyz = copy.deepcopy(pc_xyzrgb[:, 0:3])  # reserved for final visualization
    pc_xyzrgb[:, 0:1] = (pc_xyzrgb[:, 0:1] - min_x) / np.maximum((max_x - min_x), 1e-3)
    pc_xyzrgb[:, 1:2] = (pc_xyzrgb[:, 1:2] - min_y) / np.maximum((max_y - min_y), 1e-3)
    pc_xyzrgb[:, 2:3] = (pc_xyzrgb[:, 2:3] - min_z) / np.maximum((max_z - min_z), 1e-3)
    pc_xyzrgb = np.concatenate([pc_xyzrgb, ori_xyz], axis=-1)   # [points, 12]
    temp[i] = pc_xyzrgb


transformer = torch.load("J:\\COMProfile\\Downloads\\pk\\transformer_0411\\transformer_19.pth", map_location=torch.device('cpu'))
transformer.eval()
seg_net = torch.load("J:\\COMProfile\\Downloads\\pk\\transformer_0411\\seg_net_19.pth", map_location=torch.device('cpu'))
seg_net.eval()
sem_labels = np.array([], dtype=np.int32)
for i in range(0, batch, 6):
    tempX = temp[i:i+6]
    idx, points = farthest_point_sampling(tempX)
    X = torch.tensor(tempX, dtype=torch.float32)
    global_features, points_features = transformer(X, points, idx, "cpu")
    predict_sem_labels = seg_net(global_features, points_features)
    predict_sem_labels = torch.squeeze(predict_sem_labels)
    predict_sem_labels = torch.argmax(predict_sem_labels, dim=-1)   # [6, points]
    predict_sem_labels = predict_sem_labels.view(-1)                # [6 * points, ]
    predict_sem_labels = predict_sem_labels.numpy()
    sem_labels = np.concatenate((sem_labels, predict_sem_labels))

Plot.draw_pc_semins(fin['coords'][:].reshape(-1, 3), sem_labels, 13)
"""