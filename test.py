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
con = glob.glob("./data/Data_S3DIS/" + a + '*.h5')
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
fin = h5py.File("./Area_1_office_1.h5", 'r')

coords = fin['coords'][:].reshape(-1, 3)
points = fin['points'][:].reshape(-1, 9)
pc_xyzrgb = np.concatenate([coords, points[:, 3:9]], axis=-1)
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
pc_xyzrgb = np.concatenate([pc_xyzrgb, ori_xyz], axis=-1)
# print(pc_xyzrgb.shape)


backbone = torch.load("./backbone.pth", map_location=torch.device('cpu'))
backbone.eval()
seg_net = torch.load("./seg_net.pth", map_location=torch.device('cpu'))
seg_net.eval()
in_X = torch.tensor(pc_xyzrgb[None, :, :9]).clone().detach()
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