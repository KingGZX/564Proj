from helper_s3dis import Data_S3DIS
import glob
from model import Backbone_BoNet

a = 'Area_1'
con = glob.glob("./data/Data_S3DIS/" + a + '*.h5')

data = Data_S3DIS("./data/Data_S3DIS/", ['Area_1'], ['Area_5'])

X, sem_labels, ins_labels, psem_labels, bb_labels, pmask_labels = data.load_train_next_batch()
print(X.shape)

net = Backbone_BoNet(9)
import torch
gf, pf = net(torch.tensor(X[:, :, :9]))

print(gf.shape, pf.shape)
