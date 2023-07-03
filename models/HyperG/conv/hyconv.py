import torch
import numpy as np
from base_utils import *
from tqdm import tqdm
import json
import math
from torch import nn
from torch.nn import Parameter
from sklearn.preprocessing import MinMaxScaler


from models.HyperG.hyedge import degree_hyedge, degree_node, count_hyedge, count_node


class HyFMConv(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True) -> None:
        super().__init__()
        self.theta = Parameter(torch.Tensor(in_ch, out_ch))

        if bias:
            self.bias = Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def gen_hyedge_ft(self, x: torch.Tensor, H: torch.Tensor, hyedge_weight=None):
        ft_dim = x.size(1)
        node_idx, hyedge_idx = H
        hyedge_num = count_hyedge(H)

        # a vector to normalize hyperedge feature
        hyedge_norm = 1.0 / degree_hyedge(H).float()
        if hyedge_weight is not None:
            hyedge_norm *= hyedge_weight
        hyedge_norm = hyedge_norm[hyedge_idx]

        x = x[node_idx] * hyedge_norm.unsqueeze(1)
        x = torch.zeros(hyedge_num, ft_dim).to(x.device).scatter_add(0, hyedge_idx.unsqueeze(1).repeat(1, ft_dim), x)
        return x

    def gen_node_ft(self, x: torch.Tensor, H: torch.Tensor):
        ft_dim = x.size(1)
        node_idx, hyedge_idx = H
        node_num = count_node(H)

        # a vector to normalize node feature
        node_norm = 1.0 / degree_node(H).float()
        node_norm = node_norm[node_idx]

        x = x[hyedge_idx] * node_norm.unsqueeze(1)
        x = torch.zeros(node_num, ft_dim).to(x.device).scatter_add(0, node_idx.unsqueeze(1).repeat(1, ft_dim), x)
        return x

    def forward(self, x: torch.Tensor, H: torch.Tensor, hyedge_weight=None):
        assert len(x.shape) == 2, 'the input of HyperConv should be N x C'
        # feature transform
        x = x.matmul(self.theta)

        # generate hyperedge feature from node feature
        x = self.gen_hyedge_ft(x, H, hyedge_weight)

        # generate node feature from hyperedge feature
        x = self.gen_node_ft(x, H)

        if self.bias is not None:
            return x + self.bias
        else:
            return x


class HGFN_conv(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        super(HGFN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ch, out_ch))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class HyConv(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True) -> None:
        super().__init__()
        self.theta = Parameter(torch.Tensor(in_ch, out_ch))

        if bias:
            self.bias = Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def gen_hyedge_ft(self, x: torch.Tensor, H: torch.Tensor, hyedge_weight=None):
        ft_dim = x.size(1)
        node_idx, hyedge_idx = H
        hyedge_num = count_hyedge(H)

        # a vector to normalize hyperedge feature
        hyedge_norm = 1.0 / degree_hyedge(H).float()
        if hyedge_weight is not None:
            hyedge_norm *= hyedge_weight
        hyedge_norm = hyedge_norm[hyedge_idx]

        x = x[node_idx] * hyedge_norm.unsqueeze(1)
        x = torch.zeros(hyedge_num, ft_dim).to(x.device).scatter_add(0, hyedge_idx.unsqueeze(1).repeat(1, ft_dim), x)
        return x

    def gen_node_ft(self, x: torch.Tensor, H: torch.Tensor):
        ft_dim = x.size(1)
        node_idx, hyedge_idx = H
        node_num = count_node(H)

        # a vector to normalize node feature
        node_norm = 1.0 / degree_node(H).float()
        node_norm = node_norm[node_idx]

        x = x[hyedge_idx] * node_norm.unsqueeze(1)
        x = torch.zeros(node_num, ft_dim).to(x.device).scatter_add(0, node_idx.unsqueeze(1).repeat(1, ft_dim), x)
        return x

    def forward(self, x: torch.Tensor, H: torch.Tensor, hyedge_weight=None):
        assert len(x.shape) == 2, 'the input of HyperConv should be N x C'
        # feature transform
        x = x.matmul(self.theta)

        # generate hyperedge feature from node feature
        x = self.gen_hyedge_ft(x, H, hyedge_weight)

        # generate node feature from hyperedge feature
        x = self.gen_node_ft(x, H)

        if self.bias is not None:
            return x + self.bias
        else:
            return x


def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False, save_G=True, G_name=None):
    """
    calculate G from hypgraph incidence matrix H
    :param save_G:
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """

    if G_name is None:
        G_name = 'G_test'
    G_dir = os.path.join(BASE_PATH, 'graphs')
    if not os.path.exists(G_dir):
        os.makedirs(G_dir)
    G_path = os.path.join(G_dir, '{}.npy'.format(G_name))

    if not os.path.exists(G_path):
        if type(H) != list:
            G = _generate_G_from_H(H, variable_weight)
        else:
            print('Calculating G from HyperGraph!')
            G = []
            for sub_H in tqdm(H):
                G.append(generate_G_from_H(sub_H, variable_weight))

        if save_G:
            np.save(G_path, np.array(G))
            print('Save graph to: ', G_path)
    else:
        G = np.load(G_path)
    return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H


# def load_ft(feature_dir=None, save_feature_path=None, save_name='800'):
#     if feature_dir is None:
#         feature_dir = os.path.join(BASE_PATH, 'data', 'preprocess', 'features')
#
#     if save_feature_path is None:
#         save_feature_path = os.path.join(BASE_PATH, 'data', 'preprocess', 'saved_features')
#         if not os.path.exists(save_feature_path):
#             os.makedirs(save_feature_path)
#
#     concat_feature_path = os.path.join(save_feature_path, '{}_feat.npy'.format(save_name))
#     concat_labels_path = os.path.join(save_feature_path, '{}_lbls.npy'.format(save_name))
#     concat_trains_path = os.path.join(save_feature_path, '{}_train.npy'.format(save_name))
#     concat_tests_path = os.path.join(save_feature_path, '{}_test.npy'.format(save_name))
#     if os.path.exists(concat_feature_path):
#         print('Load concat feature from saved ...', concat_feature_path)
#         return np.load(concat_feature_path), np.load(concat_labels_path), \
#                np.load(concat_trains_path), np.load(concat_tests_path)
#
#     feature_concat_list = list()
#
#     if os.path.isdir(feature_dir):
#         feature_matrix = np.empty(128)
#         labels = np.empty(1, dtype=int)
#         idx_trains = list()
#         idx_tests = list()
#         print('Begin load features ... ')
#         for root, firs, files in os.walk(feature_dir):
#             for each_file in tqdm(files):
#                 if each_file.endswith('.npy'):
#                     feature_concat_list.append(each_file.split('_')[-1].split('.')[0])
#                     features, lbs, idx_train, idx_test = load_ft(os.path.join(root, each_file))
#                     feature_matrix = np.vstack((feature_matrix, features))
#                     labels = np.vstack((labels, lbs))
#                     if idx_train is not None:
#                         idx_trains.append(idx_train)
#                     if idx_test is not None:
#                         idx_tests.append(idx_test)
#         feature_matrix = feature_matrix[1:]
#         labels = labels[1:]
#         idx_trains = np.array(idx_trains)
#         idx_tests = np.array(idx_tests)
#         print('Finish load features ...', feature_matrix.shape)
#
#         with open(os.path.join(feature_dir, 'feat_concat_list.json'), 'w+') as out_f:
#             out_f.write(json.dumps(feature_concat_list))
#
#         print('Save concat features to ...', save_feature_path)
#         np.save(concat_feature_path, feature_matrix)
#         np.save(concat_labels_path, labels)
#         np.save(concat_trains_path, idx_trains)
#         np.save(concat_tests_path, idx_tests)
#
#         return feature_matrix, labels, idx_trains, idx_tests
#     else:
#         if os.path.isfile(feature_dir):
#             features = np.load(feature_dir)
#         annos = json.load(open(os.path.join(DATASET_PATH, 'opti_survival.json'), 'r'))
#         wsi_id = '{}{}'.format(class_name, str(feature_dir.split('_')[-1][0]).zfill(4))
#         labels = np.full((features.shape[0], 1), int(annos[wsi_id]))
#         split_dict = json.load(open(os.path.join(BASE_PATH, 'data', 'preprocess', 'split_simple.json'), 'r'))
#
#         idx_train = None
#         idx_val = None
#
#         if wsi_id in split_dict['train']:
#             idx_train = int(wsi_id.split('-')[-1])
#
#         if wsi_id in split_dict['val']:
#             idx_val = int(wsi_id.split('-')[-1])
#
#         return features, labels, idx_train, idx_val


# def load_feature_construct_H(features,
#                              labels,
#                              m_prob=1,
#                              K_neigs=None,
#                              is_probH=True,
#                              split_diff_scale=False,
#                              save_H=False,
#                              load_H=False,
#                              H_name=None):
#     """
#     :param feature_dir: directory of feature data
#     :param m_prob: parameter in hypergraph incidence matrix construction
#     :param K_neigs: the number of neighbor expansion
#     :param is_probH: probability Vertex-Edge matrix or binary
#     :return:
#     """
#     if H_name is None:
#         H_name = 'H_test'
#     G_dir = os.path.join(BASE_PATH, 'graphs')
#     if not os.path.exists(G_dir):
#         os.makedirs(G_dir)
#     H_path = os.path.join(G_dir, '{}.npy'.format(H_name))
#
#     if K_neigs is None:
#         K_neigs = [10]
#     # features, labels, idx_train, idx_test = load_ft(feature_dir)
#
#     if not load_H:
#         # construct hypergraph incidence matrix
#         print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
#         H = None
#         tmp = construct_H_with_KNN(features, K_neigs=K_neigs,
#                                    split_diff_scale=split_diff_scale,
#                                    is_probH=is_probH, m_prob=m_prob)
#         H = hyperedge_concat(H, tmp)
#         print('Finish constructing Hypergraph!')
#
#         if save_H:
#             np.save(H_path, H)
#             print('Save Hypergraph to:', H_path)
#     else:
#         print('load hypergraph from: ', str(H_path))
#         H = np.load(H_path)
#     # Normalize labels
#     labels = labels.reshape(-1, 1)
#     labels_shape = labels.shape
#     min_max_scaler = MinMaxScaler()
#     min_max_scaler.fit(labels)
#     labels = min_max_scaler.transform(labels)[:, 0]
#     labels = labels.reshape(labels_shape)
#
#     return H
#
#
# def predict_model(model, inputs=None, labels=None):
#     if labels is None:
#         labels = np.array(list(json.load(open(os.path.join(DATASET_PATH, 'opti_survival_test.json'), 'r')).values()))
#
#     predict_res = dict()
#     if inputs is None:
#         inputs = dict()
#         for each_feature in glob(os.path.join(DATASET_PATH, 'test_feats', '*.npy')):
#             inputs[each_feature.split('_')[-1].split('.')[0]] = np.load(each_feature)
#     # print(inputs)
#     for k, each_in in inputs.items():
#         # predict_out = model.predict(each_in)
#         predict_out = np.array([0.21])
#         min_max_scaler = MinMaxScaler()
#         min_max_scaler.fit(labels.reshape(-1, 1))
#         predict_res[k] = min_max_scaler.inverse_transform(predict_out.reshape(-1, 1))
#     return predict_res

