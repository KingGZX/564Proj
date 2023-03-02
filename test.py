from helper_s3dis import Data_S3DIS
import glob
from model import *
import torch
from lossfunc import *

"""
test the procedures step by step
"""

a = 'Area_1'
con = glob.glob("./data/Data_S3DIS/" + a + '*.h5')

data = Data_S3DIS("./data/Data_S3DIS/", ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6'], ['Area_5'])

from train import train
net = Backbone_BoNet(9)
net2 = Semantic_Net()
net3 = BBox_Net()
net4 = Pmask_Net()
train([net, net2, net3, net4], data)

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
