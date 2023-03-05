import h5py
import open3d
import numpy as np


def draw_pc():
    pc_list = list()
    data = h5py.File("Area_" + str(1) + "_office_1.h5")
    length = len(data['labels'][:])
    for i in range(length):
        coords = data['coords'][i][:]
        pc_xyzrgb = data['points'][i][:, 3:9]
        pc_xyzrgb = np.concatenate([coords, pc_xyzrgb], axis=-1)
        pc = open3d.PointCloud()
        pc.points = open3d.Vector3dVector(pc_xyzrgb[:, 0:3].reshape(-1, 3))
        if pc_xyzrgb.shape[1] == 3:
            open3d.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = open3d.Vector3dVector((pc_xyzrgb[:, 3:6] / 255.).reshape(-1, 3))
        else:
            pc.colors = open3d.Vector3dVector(pc_xyzrgb[:, 3:6].reshape(-1, 3))
        pc_list.append(pc)
    open3d.draw_geometries(pc_list)
    return 0

draw_pc()
