import torch
import torch.nn.functional as F
import torch.nn as nn

"""
Framework:
    1. self-attention layer  ------(without mask version)
    2. knn-based attention   "because directly operate on all points is time & memory consuming"
    3. FPS (farthest point sampling)
    4. all together to composite Transformer
"""


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
    indices = indices[(..., *((None,) * len(value_dims)))]              # indices [batch, points, neighbors, 1]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)  # indices [batch, points, neighbors, hidden]
    value_expand_len = len(indices_shape) - (dim + 1)                   # 3 - (2 + 1) = 0

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
            nn.ReLU(),
            nn.Linear(pos_hidden_dim, model_dim)
        )
        self.neighbors = num_neighbors
        self.attn_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * attn_hidden_dim),
            nn.ReLU(),
            nn.Linear(model_dim * attn_hidden_dim, model_dim)
        )

        self.fc = nn.Linear(model_dim, model_dim)
        self.bn1 = nn.BatchNorm1d(model_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        """
        :param x:  input point cloud in shape [batch, points_num, features]
        :return:
        """
        pos = x[:, :, -3:]  # original coordinates for calculating distances
        realx = x[:, :, :-3]  # normalized coordinates, rgb and such features
        residual = realx
        q, k, v = torch.chunk(self.to_qkv(realx), chunks=3, dim=-1)

        # shape is [batch, points, points, hidden_dim],
        # each feature map is the mapping score of one specific query and all other keys
        qk_map = q[:, :, None, :] - k[:, None, :, :]

        relative_pos = pos[:, :, None, :] - pos[:, None, :, :]
        relative_pos_embed = self.pos_embed(relative_pos)

        values_expand = v[:, None, :, :].repeat(1, x.shape[1], 1, 1)

        if self.neighbors:
            dis_norm = torch.norm(relative_pos, dim=-1)            # L2 norm to each row, in shape [b, p, p]
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
    def __init__(self):
        super(Transformer, self).__init__()
        pass

    def forward(self):
        pass


if __name__ == "__main__":
    x = torch.randn(10, 400, 12)
    sa = SA_Layer(9, 4, 64)
    out = sa(x)
    print(out.shape)