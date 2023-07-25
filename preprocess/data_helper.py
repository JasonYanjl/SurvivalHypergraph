import glob
import json
import os
import os.path as osp
import pickle
import pandas as pd

import numpy as np
import math
import torch
import itertools
import random

from preprocess.extract_patch_feature import extract_ft
from torch.utils.data import Dataset, DataLoader

from models.HyperG.utils.data import split_id
from models.HyperG.utils.data.pathology import sample_patch_coors, draw_patches_on_slide

from train_config import *

survival_time_min = 92

def split_train_val_sensored(data_root, ratio=0.8, save_split_dir=None, resplit=False, opti_survival_root = None):
    if not resplit and save_split_dir is not None and osp.exists(save_split_dir) and osp.getsize(save_split_dir) > 0:
        with open(save_split_dir, 'rb') as f:
            result = pickle.load(f)
        return result

    with open(osp.join(opti_survival_root, 'opti_survival.json'), 'r') as fp:
        lbls = json.load(fp)

    all_dict = {}
    survival_time_max = 0
    for patient in lbls.keys():
        status = lbls[patient]['status']
        time = lbls[patient]['survival_time']
        images = lbls[patient]['images']
        for image_id in images:
            img_path = data_root + '/' + image_id + '.svs'
            all_dict[image_id] = {}
            time = int(time)
            status = int(status)
            all_dict[image_id]['img_dir'] = img_path
            all_dict[image_id]['survival_time'] = time
            all_dict[image_id]['status'] = status
        survival_time_max = survival_time_max \
            if survival_time_max > time else time

    id_list = list(all_dict.keys())
    train_list, val_list = split_id(id_list, ratio)

    result = {'survival_time_max': survival_time_max,
              'train': {},
              'val': {}}
    for _id in train_list:
        result['train'][_id] = all_dict[_id]
    for _id in val_list:
        result['val'][_id] = all_dict[_id]

    if save_split_dir is not None:
        save_folder = osp.split(save_split_dir)[0]
        if not osp.exists(save_folder):
            os.makedirs(save_folder)
        with open(save_split_dir, 'wb') as f:
            pickle.dump(result, f)

    return result


def preprocess(data_dict, patch_ft_dir, patch_coors_dir, num_sample=2000,
               batch_size=256, sampled_vis=None, mini_frac=32, cnn_base='resnet', cnn_depth=34):
    # check if each slide patch feature exists
    all_dir_list = []
    for phase in ['train', 'val']:
        for _id in data_dict[phase].keys():
            all_dir_list.append(data_dict[phase][_id]['img_dir'])
    to_do_list = check_patch_ft(all_dir_list, patch_ft_dir)

    if to_do_list is not None:
        for _idx, _dir in enumerate(to_do_list):
            try:
                print(f'{_idx + 1}/{len(to_do_list)}: processing slide {_dir}...')

                print(f'sampling patch...')
                _id = get_id(_dir)
                _patch_coors = sample_patch_coors(_dir, num_sample=2000, patch_size=256)

                # save sampled patch coordinates
                with open(osp.join(patch_coors_dir, f'{_id}_coors.pkl'), 'wb') as fp:
                    pickle.dump(_patch_coors, fp)

                # visualize sampled patches on slide
                if sampled_vis is not None:
                    _vis_img_dir = osp.join(sampled_vis, f'{_id}_sampled_patches.jpg')
                    print(f'saving sampled patch_slide visualization {_vis_img_dir}...')
                    _vis_img = draw_patches_on_slide(_dir, _patch_coors, mini_frac=32)
                    with open(_vis_img_dir, 'w') as fp:
                        _vis_img.save(fp)

                # extract patch feature for each slide
                print(f'extracting feature...')
                fts = extract_ft(_dir, _patch_coors, depth=cnn_depth, batch_size=batch_size, cnn_base=cnn_base)
                np.save(osp.join(patch_ft_dir, f'{_id}_fts.npy'), fts.cpu().numpy())
            except:
                pass


def get_long_id(_dir):
    # tmp_dir = _dir[-21 :-8]
    # return tmp_dir
    tmp_dir = _dir.split('/')[-1][:-8]
    return tmp_dir

def get_dataloaders(data_dict, patch_ft_dir, batch_size=1, filter_DX=False, filter_event=False, filter_tnm=[]):
    all_ft_list = glob.glob(osp.join(patch_ft_dir, '*_fts.npy'))

    ft_dict = {}
    for _dir in all_ft_list:
        # ft_dict[get_id(_dir)] = _dir
        ft_dict[get_long_id(_dir)] = _dir

    SP_datasets = {'train': SlidePatch(data_dict['train'], ft_dict, data_dict['survival_time_max'], filter_DX=filter_DX, filter_tnm=filter_tnm),
                   'val': SlidePatch(data_dict['val'], ft_dict, data_dict['survival_time_max'], filter_DX=filter_DX, filter_event=filter_event, filter_tnm=filter_tnm),
                  }
    SP_dataloaders = {phase: DataLoader(SP_datasets[phase], batch_size=batch_size,
                                        shuffle=True, num_workers=4)
                      for phase in ['train', 'val']}
    dataset_size = {phase: len(SP_datasets[phase]) for phase in ['train', 'val']}
    # len_ft = SP_datasets['train'][0][0].size(1)
    len_ft = 521
    return SP_dataloaders, dataset_size, len_ft


def get_n_folds_dataloader(data_dict, patch_ft_dir, n=5, filter_DX=False, filter_event=False, filter_tnm=[]):
    all_ft_list = glob.glob(osp.join(patch_ft_dir, '*_fts.npy'))

    ft_dict = {}
    for _dir in all_ft_list:
        # ft_dict[get_id(_dir)] = _dir
        ft_dict[get_long_id(_dir)] = _dir

    n_fold_dataloaders = list()
    fold_length = int(math.ceil(len(all_ft_list) / float(n)))
    for i in range(0, len(all_ft_list) - fold_length, fold_length):
        val_dict = dict(itertools.islice(iter(data_dict['train'].items()), i, i + fold_length, 1))
        train_dict = {key: data_dict['train'][key] for key in data_dict['train'] if key not in val_dict}
        SP_datasets = {
            'val': SlidePatch(val_dict, ft_dict, data_dict['survival_time_max'], filter_DX=filter_DX, filter_event=filter_event, filter_tnm=filter_tnm),
            'train': SlidePatch(train_dict, ft_dict, data_dict['survival_time_max'], filter_DX=filter_DX, filter_tnm=filter_tnm)
        }
        SP_dataloaders = {phase: DataLoader(SP_datasets[phase], batch_size=1, shuffle=True, num_workers=4)
                          for phase in ['train', 'val']}
        dataset_size = {phase: len(SP_datasets[phase]) for phase in ['train', 'val']}
        # len_ft = SP_datasets['train'][0][0].size(1)
        len_ft = 521
        n_fold_dataloaders.append((SP_dataloaders, dataset_size, len_ft))
    return n_fold_dataloaders

def get_filter_tnm_list(key_list, filter_tnm):
    res = []
    preprocess_out_path = os.path.join(BASE_PATH, 'preprocess')
    id_2_ori_dict_path = os.path.join(preprocess_out_path, 'id_2_ori_dict.json')
    id_2_ori_dict = json.load(open(id_2_ori_dict_path, 'r'))

    anno_clin_crc_cell_path = os.path.join(DATA_CRC_PATH, 'clinical_info', 'clin_crc_cell.csv')
    anno_tcga_clin_path = os.path.join(DATA_CRC_PATH, 'clinical_info', 'tcga_clin_20210301.csv')
    tcga_clin_csv = pd.read_csv(anno_tcga_clin_path)
    clin_crc_cell_csv = pd.read_csv(anno_clin_crc_cell_path)

    for key in key_list:
        ori = id_2_ori_dict[key]

        now_tnm = -1

        try:
            csv_index = tcga_clin_csv['sample'].tolist().index(ori)
            now_tnm = int(tcga_clin_csv['tnm.stage'][csv_index])
        except:
            try:
                csv_index = clin_crc_cell_csv['sample'].tolist().index(ori)
                now_tnm = int(clin_crc_cell_csv['tnm.stage'][csv_index])
            except:
                continue

        if now_tnm in filter_tnm:
            res.append(key)

            if key not in data_wsi_statistic:
                data_wsi_statistic.append(key)
            if ori not in data_patient_statistic:
                data_patient_statistic.append(ori)

    return res

class SlidePatch(Dataset):
    def __init__(self, data_dict: dict, ft_dict, survival_time_max, filter_DX = False, filter_event = False, filter_tnm = []):
        super().__init__()
        self.st_max = float(survival_time_max)
        self.st_min = float(survival_time_min)

        if filter_DX == False:
            self.id_list = list(data_dict.keys())
            self.data_dict = data_dict
            self.ft_dict = ft_dict
        else:
            DX_id = []
            trans_name_path = os.path.join(DATA_CRC_PATH, 'name_list.txt')
            with open(trans_name_path, 'r') as in_f:
                for line in in_f:
                    ori, id = line.split(' -> ')
                    ori = ori.split('.svs')[0].strip()[:23]
                    id = id.split('.svs')[0].strip()

                    if ori[20: 22] == 'DX':
                        DX_id.append(id)

            self.data_dict = {}
            self.ft_dict = {}
            for _data in data_dict.keys():
                if _data in DX_id:
                    self.data_dict[_data] = data_dict[_data]
                    self.ft_dict[_data] = ft_dict[_data]
            self.id_list = list(self.data_dict.keys())

        if filter_event:
            data_dict_tmp = self.data_dict
            ft_dict_tmp = self.ft_dict
            id_list_tmp = self.id_list
            self.data_dict = {}
            self.ft_dict = {}
            self.id_list = []

            for _data in data_dict_tmp.keys():
                if data_dict_tmp[_data]['status'] == 1:
                    self.data_dict[_data] = data_dict_tmp[_data]
                    self.ft_dict[_data] = ft_dict_tmp[_data]

            self.id_list = list(self.data_dict.keys())

        if filter_tnm != []:
            data_dict_tmp = self.data_dict
            ft_dict_tmp = self.ft_dict
            id_list_tmp = self.id_list
            self.data_dict = {}
            self.ft_dict = {}
            self.id_list = []

            filter_tnm_true_list = get_filter_tnm_list(list(data_dict_tmp.keys()), filter_tnm = filter_tnm)
            for _data in data_dict_tmp.keys():
                if _data in filter_tnm_true_list:
                    self.data_dict[_data] = data_dict_tmp[_data]
                    self.ft_dict[_data] = ft_dict_tmp[_data]

            self.id_list = list(self.data_dict.keys())



    def __getitem__(self, idx: int):
        id = self.id_list[idx]
        # fts = torch.tensor(np.load(self.ft_dict[id])).float()
        st = torch.tensor(self.data_dict[id]['survival_time']).float()
        stv = st / self.st_max
        # stv = (self.st_min * (self.st_max - st)) / (st * (self.st_max - self.st_min))
        status = torch.tensor(self.data_dict[id]['status'])
        return stv, status, id

    def __len__(self) -> int:
        return len(self.id_list)


def check_patch_ft(dir_list, patch_ft_dir):
    to_do_list = []
    done_list = glob.glob(osp.join(patch_ft_dir, '*_fts.npy'))
    done_list = [get_id(_dir) for _dir in done_list]
    for _dir in dir_list:
        id = get_id(_dir)
        if id not in done_list:
            to_do_list.append(_dir)
    return to_do_list


def get_id(_dir):
    return osp.splitext(osp.split(_dir)[1])[0].split('_')[0]


# --------------------get a pair of data of any number---------------
def get_dataloaders_random_2(data_dict, patch_ft_dir, number, batch_size=1):
    all_ft_list = glob.glob(osp.join(patch_ft_dir, '*_fts.npy'))

    ft_dict = {}
    for _dir in all_ft_list:
        ft_dict[get_id(_dir)] = _dir

    SP_datasets = {phase: SlidePatchRandomTwo(data_dict[phase], ft_dict, data_dict['survival_time_max'], number)
                   for phase in ['train', 'val']}
    SP_dataloaders = {phase: DataLoader(SP_datasets[phase], batch_size=batch_size, shuffle=True, num_workers=4)
                      for phase in ['train', 'val']}
    dataset_size = {phase: len(SP_datasets[phase]) for phase in ['train', 'val']}
    len_ft = SP_datasets['train'][0][0].size(1)
    return SP_dataloaders, dataset_size, len_ft


class SlidePatchRandomTwo(Dataset):
    def __init__(self, data_dict: dict, ft_dict, survival_time_max, number):
        super().__init__()
        self.st_max = float(survival_time_max)
        self.id_list = list(data_dict.keys())
        self.data_dict = data_dict
        self.ft_dict = ft_dict
        self.number = number

    def __getitem__(self, idx: int):
        length = len(self.id_list)
        idx = idx % length
        idx_other = (random.randint(1, length - 1) + idx) % length

        id = self.id_list[idx]
        fts = torch.tensor(np.load(self.ft_dict[id])).float()
        st = torch.tensor(self.data_dict[id]['survival_time']).float()
        status = torch.tensor(self.data_dict[id]['status'])

        id_other = self.id_list[idx_other]
        fts_other = torch.tensor(np.load(self.ft_dict[id_other])).float()
        st_other = torch.tensor(self.data_dict[id_other]['survival_time']).float()
        status_other = torch.tensor(self.data_dict[id_other]['status'])

        return fts, fts_other, st / self.st_max, st_other / self.st_max, status, status_other, id, id_other

    def __len__(self) -> int:
        return self.number
