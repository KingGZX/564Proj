import glob
import numpy as np
import random
import copy
from random import shuffle
import h5py


class Data_Configs:
    # all possible semantics，instance is usually more than semantics
    sem_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
                 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
    sem_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    points_cc = 9                   # point feature dimension which is same as shape[1] of attribute ['point']
    sem_num = len(sem_names)        # semantic num
    ins_max_num = 24                # maximum instance num we assume in a batch of file
    train_pts_num = 4096
    test_pts_num = 4096


class Data_S3DIS:
    def __init__(self, dataset_path, train_areas, test_areas, train_batch_size=4):
        self.root_folder_4_train_test = dataset_path
        self.train_files = self.load_full_file_list(areas=train_areas)
        self.test_files = self.load_full_file_list(areas=test_areas)
        print('train files:', len(self.train_files))
        print('test files:', len(self.test_files))

        self.ins_max_num = Data_Configs.ins_max_num  # used in bounding box regression
        self.train_batch_size = train_batch_size
        self.test_batch_size = train_batch_size
        self.total_train_batch_num = len(self.train_files) // self.train_batch_size
        self.total_test_batch_num = len(self.test_files) // self.test_batch_size


        self.train_next_bat_index = 0
        self.test_next_bat_index = 0

    def load_full_file_list(self, areas):
        """
        convert original files into blocks of data
        :param areas
        :return:
        """
        all_files = []
        for a in areas:
            print('check area:', a)
            files = sorted(glob.glob(self.root_folder_4_train_test + a + '*.h5'))  # Area_x*.h5
            for f in files:
                fin = h5py.File(f, 'r')
                coords = fin['coords'][:]  # [97, 4096, 3]
                semIns_labels = fin['labels'][:].reshape([-1, 2])
                ins_labels = semIns_labels[:, 1]  # instance label
                sem_labels = semIns_labels[:, 0]  # semantic label

                valid = True
                ins_idx = np.unique(ins_labels)  # take out all possible instances
                for i_i in ins_idx:
                    if i_i <= -1:  # invalid
                        continue
                    """
                    all the points that belongs to the same instance must have same sematic label 
                    if we find anomaly, abandon it
                    """
                    sem_labels_tp = sem_labels[ins_labels == i_i]
                    unique_sem_labels = np.unique(sem_labels_tp)
                    if len(unique_sem_labels) >= 2:
                        print('>= 2 sem for an ins:', f)
                        valid = False
                        break
                if not valid:
                    continue  # ignore this file
                block_num = coords.shape[0]  # 97
                for b in range(block_num):
                    all_files.append(f + '_' + str(b).zfill(4))

        return all_files

    def load_train_next_batch(self):
        """
        important, load batch of data for training
        :return: train data, semantic labels, instance labels, point_semantic_onehot_label, bound box, point mask
        """
        bat_files = self.train_files[self.train_next_bat_index * self.train_batch_size:(
                                                    self.train_next_bat_index + 1) * self.train_batch_size]
        bat_pc = []
        bat_sem_labels = []
        bat_ins_labels = []
        bat_psem_onehot_labels = []
        bat_bbvert_padded_labels = []
        bat_pmask_padded_labels = []
        for file in bat_files:      # block by block
            pc, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = \
                Data_S3DIS.load_fixed_points(file)
            bat_pc.append(pc)
            bat_sem_labels.append(sem_labels)
            bat_ins_labels.append(ins_labels)
            bat_psem_onehot_labels.append(psem_onehot_labels)
            bat_bbvert_padded_labels.append(bbvert_padded_labels)
            bat_pmask_padded_labels.append(pmask_padded_labels)

        bat_pc = np.asarray(bat_pc, dtype=np.float32)
        bat_sem_labels = np.asarray(bat_sem_labels, dtype=np.float32)
        bat_ins_labels = np.asarray(bat_ins_labels, dtype=np.float32)
        bat_psem_onehot_labels = np.asarray(bat_psem_onehot_labels, dtype=np.float32)
        bat_bbvert_padded_labels = np.asarray(bat_bbvert_padded_labels, dtype=np.float32)
        bat_pmask_padded_labels = np.asarray(bat_pmask_padded_labels, dtype=np.float32)

        self.train_next_bat_index += 1
        return bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot_labels, bat_bbvert_padded_labels, \
            bat_pmask_padded_labels

    def load_test_next_batch_random(self):
        idx = random.sample(range(len(self.test_files)), self.train_batch_size)
        bat_pc = []
        bat_sem_labels = []
        bat_ins_labels = []
        bat_psem_onehot_labels = []
        bat_bbvert_padded_labels = []
        bat_pmask_padded_labels = []
        for i in idx:
            file = self.test_files[i]
            pc, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = \
                Data_S3DIS.load_fixed_points(file)
            bat_pc.append(pc)
            bat_sem_labels.append(sem_labels)
            bat_ins_labels.append(ins_labels)
            bat_psem_onehot_labels.append(psem_onehot_labels)
            bat_bbvert_padded_labels.append(bbvert_padded_labels)
            bat_pmask_padded_labels.append(pmask_padded_labels)

        bat_pc = np.asarray(bat_pc, dtype=np.float32)
        bat_sem_labels = np.asarray(bat_sem_labels, dtype=np.float32)
        bat_ins_labels = np.asarray(bat_ins_labels, dtype=np.float32)
        bat_psem_onehot_labels = np.asarray(bat_psem_onehot_labels, dtype=np.float32)
        bat_bbvert_padded_labels = np.asarray(bat_bbvert_padded_labels, dtype=np.float32)
        bat_pmask_padded_labels = np.asarray(bat_pmask_padded_labels, dtype=np.float32)

        return bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot_labels, bat_bbvert_padded_labels, \
            bat_pmask_padded_labels

    def shuffle_train_files(self, ep):
        """
        kind of data augmentation
        """
        index = list(range(len(self.train_files)))
        random.seed(ep)
        shuffle(index)  # return a permutation of original ordered list
        train_files_new = []
        for i in index:
            train_files_new.append(self.train_files[i])
        self.train_files = train_files_new
        self.train_next_bat_index = 0
        print('train files shuffled!')

    def load_test_next_batch_sq(self):
        bat_files = self.test_files[self.test_next_bat_index * self.test_batch_size:(self.test_next_bat_index + 1) * self.test_batch_size]
        bat_pc = []
        bat_sem_labels = []
        bat_ins_labels = []
        bat_psem_onehot_labels = []
        bat_bbvert_padded_labels = []
        bat_pmask_padded_labels = []
        for file in bat_files:
            pc, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = \
                Data_S3DIS.load_fixed_points(file)
            bat_pc += [pc]
            bat_sem_labels += [sem_labels]
            bat_ins_labels += [ins_labels]
            bat_psem_onehot_labels += [psem_onehot_labels]
            bat_bbvert_padded_labels += [bbvert_padded_labels]
            bat_pmask_padded_labels += [pmask_padded_labels]

        bat_pc = np.asarray(bat_pc, dtype=np.float32)
        bat_sem_labels = np.asarray(bat_sem_labels, dtype=np.float32)
        bat_ins_labels = np.asarray(bat_ins_labels, dtype=np.float32)
        bat_psem_onehot_labels = np.asarray(bat_psem_onehot_labels, dtype=np.float32)
        bat_bbvert_padded_labels = np.asarray(bat_bbvert_padded_labels, dtype=np.float32)
        bat_pmask_padded_labels = np.asarray(bat_pmask_padded_labels, dtype=np.float32)

        self.test_next_bat_index += 1
        return bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot_labels, bat_bbvert_padded_labels, \
            bat_pmask_padded_labels, bat_files

    @staticmethod
    def load_fixed_points(file_path):
        """
        :param file_path:
        :return:
        """
        pc_xyzrgb, sem_labels, ins_labels = Data_S3DIS.load_raw_data_file_s3dis_block(file_path)

        # find the rectangular which includes the block
        min_x = np.min(pc_xyzrgb[:, 0])
        max_x = np.max(pc_xyzrgb[:, 0])
        min_y = np.min(pc_xyzrgb[:, 1])
        max_y = np.max(pc_xyzrgb[:, 1])
        min_z = np.min(pc_xyzrgb[:, 2])
        max_z = np.max(pc_xyzrgb[:, 2])

        ori_xyz = copy.deepcopy(pc_xyzrgb[:, 0:3])  # reserved for final visualization

        # normalize the coordinate
        use_zero_one_center = True
        if use_zero_one_center:
            pc_xyzrgb[:, 0:1] = (pc_xyzrgb[:, 0:1] - min_x) / np.maximum((max_x - min_x), 1e-3)
            pc_xyzrgb[:, 1:2] = (pc_xyzrgb[:, 1:2] - min_y) / np.maximum((max_y - min_y), 1e-3)
            pc_xyzrgb[:, 2:3] = (pc_xyzrgb[:, 2:3] - min_z) / np.maximum((max_z - min_z), 1e-3)

        pc_xyzrgb = np.concatenate([pc_xyzrgb, ori_xyz], axis=-1)

        # turn in to 4096 dimensions vector
        sem_labels = sem_labels.reshape([-1])
        ins_labels = ins_labels.reshape([-1])
        bbvert_padded_labels, pmask_padded_labels = Data_S3DIS.get_bbvert_pmask_labels(pc_xyzrgb, ins_labels)

        psem_onehot_labels = np.zeros((pc_xyzrgb.shape[0], Data_Configs.sem_num), dtype=np.int8)
        for idx, s in enumerate(sem_labels):
            # idx is the point index while s is the semantic label
            if sem_labels[idx] == -1:
                continue  # invalid points
            sem_idx = Data_Configs.sem_ids.index(s)
            psem_onehot_labels[idx, sem_idx] = 1

        """
        psem_onehot_labels is a matrix, row is the number of points, column is number of semantics. 
        so we set psem_onehot_labels[x, y] = 1 if point x's semantic label is y
        
        bbvert_padded_labels is a dimension 3 matrix:
            dimension 1: maximum number of instances    (actually, one block may not have so many instances)
            dimension 2 and 3 consists of [2, 3] which is a representation of rectangular. 
            in the form [[x_min, y_min, z_min],
                        [x_max, y_max, z_max]].
                        
        pmask_padded_labels is a matrix, row is the number of maximum number of instances, column is number of points.
        so we set pmask_padded_labels[x, y] = 1 if point y belongs to instance x
            
        """

        return pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels

    @staticmethod
    def load_raw_data_file_s3dis_block(file_path):
        # our raw data in one .h5 file is [blocks, 4096, dim] dim depends on attribute
        block_id = int(file_path[-4:])
        file_path = file_path[0:-5]

        fin = h5py.File(file_path, 'r')
        coords = fin['coords'][block_id]  # [4096, 3]
        points = fin['points'][block_id]  # [4096, 9]
        semIns_labels = fin['labels'][block_id]  # [4096, 2]

        """
        confused about what points attribute means
        """

        pc = np.concatenate([coords, points[:, 3:9]], axis=-1)
        sem_labels = semIns_labels[:, 0]
        ins_labels = semIns_labels[:, 1]

        # if you need to visualize data, uncomment the following lines
        """
        from helper_data_plot import Plot as Plot
        Plot.draw_pc(pc)
        Plot.draw_pc_semins(pc_xyz=pc[:, 0:3], pc_semins=sem_labels, fix_color_num=13)
        Plot.draw_pc_semins(pc_xyz=pc[:, 0:3], pc_semins=ins_labels)
        """

        return pc, sem_labels, ins_labels

    @staticmethod
    def get_bbvert_pmask_labels(pc, ins_labels):
        """
        :param pc:  x, y, z, r, g, b, p, q, w, norm_x, norm_y, norm_z
        :param ins_labels: a (4096,) vector
        :return:
        """

        # gt: ground_truth?   bounding boxs
        gt_bbvert_padded = np.zeros((Data_Configs.ins_max_num, 2, 3), dtype=np.float32)
        # it's like, we will assign a value of a point depends on whether it's in an instance
        gt_pmask = np.zeros((Data_Configs.ins_max_num, pc.shape[0]), dtype=np.float32)
        count = -1
        unique_ins_labels = np.unique(ins_labels)
        for ins_ind in unique_ins_labels:
            if ins_ind <= -1:
                continue
            count += 1
            if count >= Data_Configs.ins_max_num:
                print('ignored! more than max instances:', len(unique_ins_labels))
                continue

            ins_labels_tp = np.zeros(ins_labels.shape, dtype=np.int8)
            ins_labels_tp[ins_labels == ins_ind] = 1
            ins_labels_tp = np.reshape(ins_labels_tp, [-1])         # to a vector
            gt_pmask[count, :] = ins_labels_tp

            ins_labels_tp_ind = np.argwhere(ins_labels_tp == 1)
            ins_labels_tp_ind = np.reshape(ins_labels_tp_ind, [-1])     # to a vector

            # bb min_xyz, max_xyz
            pc_xyz_tp = pc[:, 0:3]
            pc_xyz_tp = pc_xyz_tp[ins_labels_tp_ind]    # take all points' coordinate which belongs to instance ins_ind
            """
            setting gt_bbvert_padded, we can get the bounding box of instance
            """
            gt_bbvert_padded[count, 0, 0] = np.min(pc_xyz_tp[:, 0])
            gt_bbvert_padded[count, 0, 1] = np.min(pc_xyz_tp[:, 1])
            gt_bbvert_padded[count, 0, 2] = np.min(pc_xyz_tp[:, 2])
            gt_bbvert_padded[count, 1, 0] = np.max(pc_xyz_tp[:, 0])
            gt_bbvert_padded[count, 1, 1] = np.max(pc_xyz_tp[:, 1])
            gt_bbvert_padded[count, 1, 2] = np.max(pc_xyz_tp[:, 2])

        return gt_bbvert_padded, gt_pmask
