import os
import torch
from pprint import pprint

from base_utils import *

k_nearest = 9
model_name = 'hgrcox'  # None
ignore_censored = False

batch_size = 512
mini_frac = 32
n_target = 1
num_epochs = 300

sensitive = 'attribute'  # attribute, pattern
pooling_strategy = 'max'  # mean, max
learning_rate = 0.0005  # 0.0001

dropout = 0.3
dropmax_ratio = 0

# ------------ probability -----------------
repeat = 30  # generate probability during training
S_top = 5  # select top 5 best samples in number of repeat
MODE = 'top_S'  # top_S, whole

if sensitive in 'pattern':
    hiddens = [2000, 2000]
if sensitive in 'attribute':
    hiddens = [128, 128]

cnn_base = 'resnet'  # ['resnet', 'vgg']
cnn_depth = 18      # 18 -> 512, 34.. -> 2048

optimizer = 'sgd'  # adam, sgd
trainset = 'all'  # all, samples

filter_DX = False
filter_event = False
filter_tnm = []

data_patient_statistic = []
data_wsi_statistic = []

# ------------ for fm --------------------
field_dim = [512]
embed_dim = 1000   # omiga
mm_phi = 400

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_CRC_PATH = '/data/home/dl/'

dataset_name = 'CRC-SYSU-SAH-HE'

def get_config(pick_dataset=None):
    if pick_dataset.upper() == 'CRC-SYSU-SAH-HE':
        data_root = os.path.join(BASE_PATH, 'data')
        result_root = os.path.join(BASE_PATH, 'result', 'result')
        roc_km_output_save_path = os.path.join(BASE_PATH, 'result', 'roc_km_output')

    else:
        exit(0)

    svs_dir = result_root
    patch_ft_dir = os.path.join(data_root, 'CRC-SYSU-SAH-HE-feature')

    sampled_vis = os.path.join(result_root, 'sampled_vis')
    patch_coors_dir = os.path.join(result_root, 'patch_coors')

    split_dir = os.path.join(result_root, 'split.pkl')

    model_save_path = os.path.join(result_root, 'model_best_{}.pth'.format(model_name))

    checkpoint_dir = os.path.join(result_root, 'checkpoint')

    if not os.path.exists(roc_km_output_save_path):
        os.makedirs(roc_km_output_save_path)

    return data_root, result_root, svs_dir, patch_coors_dir, sampled_vis, patch_ft_dir, split_dir, \
           model_save_path, checkpoint_dir, roc_km_output_save_path


def show_configs(**configs):
    def parastr(obj, namespace):
        return [name for name in namespace if namespace[name] is obj]

    for cfg in (num_sample, batch_size, mini_frac, num_epochs, sensitive,
                pooling_strategy, dropout, dropmax_ratio, hiddens, cnn_base, cnn_depth,
                optimizer, trainset, device):
        configs[parastr(cfg, globals())[-1]] = cfg
    pprint(configs)
