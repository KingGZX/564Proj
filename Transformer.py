import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

"""
Framework:
    1. self-attention layer  ------(without mask version)
    2. knn-based attention   "because directly operate on all points is time & memory consuming"
    3. FPS (farthest point sampling) & interpolation
    4. all together to composite Transformer
"""


def farthest_point_sampling(points, sample_num=500):
    """
    :param points:        [Batch, points, features]
    :param sample_num:    down sample num
    :return:              index = [Batch, sample_num]     sample_points = [Batch, sample_num, features]
    """
    batch, num, fd = points.shape
    select_idx = np.zeros((batch, sample_num))
    select_idx[np.arange(batch)] = np.arange(sample_num)
    if num > sample_num:
        coords = points[:, :, -3:]  # the last 3 dimensions are original coordinates
        features = points[:, :, :-3]
        select_index = np.random.randint(0, points.shape[1], size=batch)
        distances = np.zeros((batch, num)) + 1e10
        for i in range(sample_num):
            select_idx[:, i] = select_index
            select_coords = coords[np.arange(batch), select_index, :]
            select_coords = select_coords.reshape((-1, 1, 3))
            # [batch, num, 3]
            select_coords = np.repeat(select_coords, num, axis=1)
            new_distances = np.linalg.norm(select_coords - coords, axis=-1)
            idx = new_distances < distances
            distances[idx] = new_distances[idx]
            select_index = np.argmax(distances, axis=1)
        """
        given select indexes [Batch, sample_num]
        fetch corresponding features from original points data, return as [Batch, sample_num, features]
        """
        index = torch.tensor(select_idx, dtype=torch.int64)
        points = torch.tensor(points)
        index = index[:, :, None]
        index = index.repeat(1, 1, fd)
        return select_idx, torch.gather(input=points, dim=1, index=index)
    else:
        """
        needs to be modified!
        """
        return select_idx, torch.tensor(points)


def neighbor_select(values, indices, dim=2):
    """
    :param values:
    Attention map , in shape [Batch, points_num, points_num, dim]
    relative_pos_embedding, [Batch, points_num, points_num, pos_embed]
    Value, ...
    :param indices: in shape [Batch, points_num, k_neighbor]
    :param dim:
    :return:
    """
    # the last dimension of every possible value's feature map(dim=2) , it's just hidden_dim = 128
    value_dims = values.shape[(dim + 1):]

    # convert the shape from torch.Size into List
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))

    """
    Explanation: 
        len(value_dims) = 1
        (None,) is a tuple, repeat n times when multiply a integer n
        -----e.g. (None, ) * 2 = (None, None)-----
        
        use * to unpack , thus, indices = indices[:, :, :, None]
        '...' stands for all dimensions of indices
        
        the main idea is to make the dimension of indices same as values and expand the feature dimension to
        make use of 'gather' API
    """
    indices = indices[(..., *((None,) * len(value_dims)))]  # indices [batch, points, neighbors, 1]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)  # indices [batch, points, neighbors, hidden]
    value_expand_len = len(indices_shape) - (dim + 1)  # 3 - (2 + 1) = 0

    #
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


class SA_Layer(nn.Module):
    def __init__(self, model_dim, attn_hidden_dim, pos_hidden_dim, num_neighbors=10):
        super(SA_Layer, self).__init__()
        self.to_qkv = nn.Linear(model_dim, model_dim * 3)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, pos_hidden_dim),  # 3D coordinates -> hidden_dim
            nn.ReLU(inplace=True),
            nn.Linear(pos_hidden_dim, model_dim)
        )
        self.neighbors = num_neighbors
        self.attn_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * attn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim * attn_hidden_dim, model_dim)
        )

        self.fc = nn.Linear(model_dim, model_dim)
        self.bn1 = nn.BatchNorm1d(model_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, pos):
        """
        :param x:  input point cloud in shape [batch, points_num, features]
        :return:
        """

        """
        pos = x[:, :, -3:]  # original coordinates for calculating distances
        realx = x[:, :, :-3]  # normalized coordinates, rgb and such features
        """

        realx = x
        residual = realx
        q, k, v = torch.chunk(self.to_qkv(realx), chunks=3, dim=-1)

        # shape is [batch, points, points, hidden_dim],
        # each feature map is the mapping score of one specific query and all other keys
        qk_map = q[:, :, None, :] - k[:, None, :, :]

        # pos = pos.to(device)
        # self.pos_embed = self.pos_embed.to(device)
        relative_pos = pos[:, :, None, :] - pos[:, None, :, :]
        relative_pos_embed = self.pos_embed(relative_pos)

        """
        relative_pos_embed = relative_pos_embed.cpu()
        relative_pos = relative_pos.cpu()
        torch.cuda.empty_cache()
        """

        values_expand = v[:, None, :, :].repeat(1, x.shape[1], 1, 1)

        if self.neighbors:
            dis_norm = torch.norm(relative_pos, dim=-1)  # L2 norm to each row, in shape [b, p, p]
            # use knn to select nearest points
            values, indices = torch.topk(dis_norm, self.neighbors, dim=-1, largest=False)

            """
            after knn sampling, all the following values are in shape [batch, points, neighbors, hidden_dim]
            """
            qk_map = neighbor_select(qk_map, indices)
            relative_pos_embed = neighbor_select(relative_pos_embed, indices)
            values_expand = neighbor_select(values_expand, indices)

        scores = self.attn_mlp(qk_map + relative_pos_embed)
        # since scores are in shape [batch, points, neighbors, hidden_dim]
        # for one feature_map, we should use softmax along column axis
        # then when we use Hadamard, it's the linear combinations of neighbor value vectors
        attn = F.softmax(scores, dim=-2)
        agg = torch.sum(attn * values_expand, dim=-2)

        # use GNN Laplacian Matrix idea
        agg = self.fc(agg)
        agg = torch.permute(agg, [0, 2, 1])  # to become  [batch, channels / features, points]
        agg = self.bn1(agg)
        agg = self.act(agg)
        agg = torch.permute(agg, [0, 2, 1])  # back to shape [batch, points, features]
        return agg + residual


class Transformer(nn.Module):
    def __init__(self, model_dim, attn_hidden_dim, pos_hidden_dim):
        super(Transformer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.attn1 = SA_Layer(64, attn_hidden_dim, pos_hidden_dim)
        self.attn2 = SA_Layer(64, attn_hidden_dim, pos_hidden_dim)
        self.attn3 = SA_Layer(64, attn_hidden_dim, pos_hidden_dim)
        self.attn4 = SA_Layer(64, attn_hidden_dim, pos_hidden_dim)
        self.attn_mlp = nn.Sequential(nn.Linear(4 * 64, 64),
                                      nn.LeakyReLU(inplace=True),
                                      )
        self.conv_fuse = nn.Sequential(nn.Conv1d(64, 512, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(512),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(inplace=True),
                                       )

    def forward(self, x, fps_points, fps_index, device="cuda"):
        """
        :param x: input data, [batch, fps_sample_num, model_dim]
        :return:  like backbone_net, get global features and point-wise features
        """
        # mapping to a high dimension feature space
        pos = x[:, :, -3:]
        x = x[:, :, :-3]
        x = torch.permute(x, [0, 2, 1])  # [batch, model_dim, num]
        f1 = self.relu1(self.bn1(self.conv1(x)))  # shape is [batch, 32, num]
        f2 = self.relu2(self.bn2(self.conv2(f1)))  # shape is [batch, 64, num]
        f2 = torch.permute(f2, [0, 2, 1])  # shape is [batch, num, 64]

        # for self-attention, shape of input and output is the same
        # to save gpu memory, only use fps samples
        fps_index = torch.tensor(fps_index, dtype=torch.int64)
        fps_index = fps_index.to(device)
        index = fps_index[:, :, None]
        index = index.repeat(1, 1, f2.shape[-1])
        fps_f2 = torch.gather(input=f2, dim=1, index=index)
        fps_pos = fps_points[:, :, -3:]
        sa1 = self.attn1(fps_f2, fps_pos)  # [batch, sample_num, 128]
        """
        sa1c = sa1.cpu()
        del sa1
        torch.cuda.empty_cache()
        """
        sa2 = self.attn2(fps_f2, fps_pos)
        """
        sa2c = sa2.cpu()
        del sa2
        torch.cuda.empty_cache()
        """
        sa3 = self.attn3(fps_f2, fps_pos)
        """
        sa3c = sa3.cpu()
        del sa3
        torch.cuda.empty_cache()
        """
        sa4 = self.attn4(fps_f2, fps_pos)
        """
        sa4c = sa4.cpu()
        del sa4
        torch.cuda.empty_cache()
        """
        attn = torch.concat((sa1, sa2, sa3, sa4), dim=2)  # [batch, sample_num, 512]

        # attn = attn.to(device)
        attn = self.attn_mlp(attn)

        attention = torch.zeros_like(f2)
        batch = x.shape[0]
        for i in range(batch):
            attention[i, fps_index[i]] = attn[i]

        attention = f2 + attention
        # to a higher space
        attention = torch.permute(attention, [0, 2, 1])
        features = self.conv_fuse(attention)
        p_feature = torch.permute(features, [0, 2, 1])  # [batch, points_num, 1024]
        g_feature = torch.max(features, dim=2)[0]
        return g_feature, p_feature


"""
if __name__ == "__main__":
    x = torch.randn(4, 4000, 12)
    temp = x.numpy()
    idx, points = farthest_point_sampling(temp)
    model = Transformer(12 - 3, 4, 64)
    out = model(x, points, idx)
    print(out.shape)
"""
