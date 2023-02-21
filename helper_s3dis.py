import glob
import numpy as np
import random
import copy
from random import shuffle
import h5py


class Data_Configs:
    # all possible semantics
    sem_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
                 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
    sem_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    points_cc = 9
    sem_num = len(sem_names)
    ins_max_num = 24
    train_pts_num = 4096
    test_pts_num = 4096


class Data_S3DIS:
    def __init__(self, dataset_path, train_areas, test_areas, train_batch_size=4):
        self.root_folder_4_traintest = dataset_path
        self.train_files = self.load_full_file_list(areas=train_areas)
        self.test_files = self.load_full_file_list(areas=test_areas)
        print('train files:', len(self.train_files))
        print('test files:', len(self.test_files))

        self.ins_max_num = Data_Configs.ins_max_num
        self.train_batch_size = train_batch_size
        self.total_train_batch_num = len(self.train_files) // self.train_batch_size

        self.train_next_bat_index = 0

    def load_full_file_list(self, areas):
        all_files = []
        for a in areas:
            print('check area:', a)
            files = sorted(glob.glob(self.root_folder_4_traintest + a + '*.h5'))  # Area_x*.h5
            for f in files:
                fin = h5py.File(f, 'r')
                coords = fin['coords'][:]           # [97, 4096, 3]
                semIns_labels = fin['labels'][:].reshape([-1, 2])
                ins_labels = semIns_labels[:, 1]  # instance label
                sem_labels = semIns_labels[:, 0]  # semantic label

                data_valid = True
                ins_idx = np.unique(ins_labels)  # take out all possible instances
                for i_i in ins_idx:
                    if i_i <= -1:   # invalid
                        continue
                    """
                    all the points that belongs to the same instance must have same sematic label 
                    """
                    sem_labels_tp = sem_labels[ins_labels == i_i]
                    unique_sem_labels = np.unique(sem_labels_tp)
                    if len(unique_sem_labels) >= 2:
                        print('>= 2 sem for an ins:', f)
                        data_valid = False
                        break
                if not data_valid:
                    continue
                block_num = coords.shape[0]     # 97
                for b in range(block_num):
                    all_files.append(f + '_' + str(b).zfill(4))

        return all_files

