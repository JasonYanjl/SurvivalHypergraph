from itertools import product

import torch

from . import remove_negative_index, self_loop_add, pairwise_euclidean_distance


def neighbor_grid(input_size, self_loop=False, neigh_funs__mask_funs=None):
    """

    :param input_size: w \times h matrix
    :param self_loop:
    :return:
    """
    assert len(input_size) == 2, \
        f"grid neighbor input size error, must be a matrix with two dimensions"
    node_num = input_size[0] * input_size[1]
    node_set = torch.arange(node_num)

    if neigh_funs__mask_funs is None:
        neigh_funs = [
            lambda _idx, _w, _h: _idx - _w - 1,
            lambda _idx, _w, _h: _idx - _w,
            lambda _idx, _w, _h: _idx - _w + 1,

            lambda _idx, _w, _h: _idx - 1,
            lambda _idx, _w, _h: _idx + 1,

            lambda _idx, _w, _h: _idx + _w - 1,
            lambda _idx, _w, _h: _idx + _w,
            lambda _idx, _w, _h: _idx + _w + 1,
        ]
        neigh_mask_funs = [
            lambda _w, _h: (node_set // _w == 0) | (node_set % _w == 0),
            lambda _w, _h: (node_set // _w == 0),
            lambda _w, _h: (node_set // _w == 0) | (node_set % _w == _w - 1),

            lambda _w, _h: (node_set % _w == 0),
            lambda _w, _h: (node_set % _w == _w - 1),

            lambda _w, _h: (node_set % _w == 0) | (node_set // _w == _h - 1),
            lambda _w, _h: (node_set // _w == _h - 1),
            lambda _w, _h: (node_set // _w == _h - 1) | (node_set % _w == _w - 1),
        ]
    else:
        neigh_funs, neigh_mask_funs = neigh_funs__mask_funs

    neigh_masks = [neigh_mask_fun(input_size[0], input_size[1]) for neigh_mask_fun in neigh_mask_funs]
    neigh_num = len(neigh_funs)

    node_idx = []
    for neigh_fun, neigh_mask in zip(neigh_funs, neigh_masks):
        _tmp = neigh_fun(node_set, input_size[0], input_size[1])
        _tmp[neigh_mask] = -1
        node_idx.append(_tmp)
    # neigh_num * input_size
    node_idx = torch.stack(node_idx, dim=1).reshape(-1)

    hyedge_idx = node_set.unsqueeze(0).repeat(neigh_num, 1).transpose(1, 0).reshape(-1)

    # construct sparse hypergraph adjacency matrix from (node_idx,hyedge_idx) pair.
    H = torch.stack([node_idx, hyedge_idx])

    H = remove_negative_index(H)

    if self_loop:
        H = self_loop_add(H)

    return H


def neighbor_distance(x: torch.Tensor, k_nearest, dis_metric=pairwise_euclidean_distance, is_cross_WSI=False):
    """
    construct hyperedge for each node in x matrix. Each hyperedge contains a node and its k-1 nearest neighbors.
    :param x: N x C matrix. N denotes node number, and C is the feature dimension.
    :param k_nearest:
    :return:
    """

    assert len(x.shape) == 2, 'should be a tensor with dimension (N x C)'

    # N x C
    node_num = x.size(0)
    dis_matrix1 = dis_metric(x)
    _, nn_idx1 = torch.topk(dis_matrix1, k_nearest, dim=1, largest=False)
    hyedge_idx1 = torch.arange(node_num).to(x.device).unsqueeze(0).repeat(k_nearest, 1).transpose(1, 0).reshape(-1)
    H1 = torch.stack([nn_idx1.reshape(-1), hyedge_idx1])
    H = H1

    if is_cross_WSI:
        dis_matrix2 = dis_metric(x)
        tot_wsi = int(x.shape[0] / 4000)
        for wsi in range(tot_wsi):
            dis_matrix2[(4000 * wsi): (4000 * (wsi + 1)), (4000 * wsi): (4000 * (wsi + 1))] = float('inf')
        torch.diagonal(dis_matrix2).fill_(0)
        _, nn_idx2 = torch.topk(dis_matrix2, k_nearest, dim=1, largest=False)
        hyedge_idx2 = torch.arange(node_num).to(x.device).unsqueeze(0).repeat(k_nearest, 1).transpose(1, 0).reshape(-1)
        hyedge_idx2 = hyedge_idx2 + node_num
        H2 = torch.stack([nn_idx2.reshape(-1), hyedge_idx2])
        H = torch.concat([H1, H2], dim=1)
    return H


def neighbor_distance_graph(x: torch.Tensor, k_nearest, dis_metric=pairwise_euclidean_distance):
    assert len(x.shape) == 2, 'should be a tensor with dimension (N x C)'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # N x C
    node_num = x.size(0)
    dis_matrix1 = dis_metric(x)
    _, nn_idx1 = torch.topk(dis_matrix1, k_nearest, dim=1, largest=False)
    res = torch.zeros(node_num, node_num, device=device).scatter(1, nn_idx1,
                                                                  torch.ones(node_num, node_num, device=device))

    res = (res + res.T) / 2
    DV = torch.sum(res, axis=1)
    DV2 = torch.diag(torch.pow(DV, -0.5))
    res = torch.matmul(DV2, torch.matmul(res, DV2))
    return res


def gather_patch_ft(x: torch.Tensor, patch_size):
    """

    :param x: 1 x C x M x N
    :param patch_size: row x column
    :return:
    """
    assert len(x.shape) == 4
    assert len(patch_size) == 2

    # C x M x N
    x = x.float().view([x.size(1), x.size(2), x.size(3)])
    x_row_num, x_col_num = x.shape[1], x.shape[2]
    # C x M x N -> M x N x C -> MN x C
    x = x.permute([1, 2, 0])
    # M x N x C -> MN x C
    x = x.view(-1, x.size(2))
    # MN x C -> (1 + MN) x C
    x = torch.cat([torch.zeros(x.size(1)).unsqueeze(0), x])

    # generate out index
    out_idx = []
    center_row, center_col = (patch_size[0] + 1) // 2 - 1, (patch_size[1] + 1) // 2 - 1
    x_idx = torch.arange(x_row_num * x_col_num).view(x_row_num, x_col_num).long()

    x_idx_pad = torch.zeros(x_row_num + patch_size[0] - 1, x_col_num + patch_size[1] - 1).long()
    x_idx_pad[center_row:center_row + x_row_num, center_col:center_col + x_col_num] = x_idx + 1

    for _row, _col in product(range(patch_size[0]), range(patch_size[1])):
        out_idx.append(x_idx_pad[_row:_row + x_row_num, _col:_col + x_col_num].reshape(-1, 1))
    # MN x kk
    out_idx = torch.cat(out_idx, dim=1)

    # apply out index
    # MNkk x C
    out = x[out_idx.view(-1)]
    # M x N x kkC
    out = out.view(x_row_num, x_col_num, -1)
    # 1 x kkC x M x N
    out = out.permute([2, 0, 1]).unsqueeze(0)
    return out
