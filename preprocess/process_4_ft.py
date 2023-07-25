import os
import sys
import random
import pandas as pd

import torchvision.models as models

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from base_utils import *
import json
from rich.progress import track
from pprint import pprint

from preprocess.generate_patch import *
from preprocess.extract_patch_feature import *

import time

DATA_CRC_PATH = os.path.join(BASE_PATH, 'svs_directory')

svs_path = DATA_CRC_PATH

feature_save_path = os.path.join(BASE_PATH, 'data', 'CRC-SYSU-SAH-HE-feature')

if not os.path.exists(feature_save_path):
    os.makedirs(feature_save_path)

preprocess_out_path = os.path.join(BASE_PATH, 'preprocess')
if not os.path.exists(preprocess_out_path):
    os.makedirs(preprocess_out_path)


def load_model(pretrained):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(in_features=512, out_features=9, bias=True)
    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
analysis_model = None

model_path = os.path.join(BASE_PATH, 'preprocess', 'model-resnet18-201103-best.pth')
analysis_model = load_model(pretrained=False).to(device)
analysis_model.load_state_dict(torch.load(model_path))
analysis_model = analysis_model.eval()


class TissueDataset(Dataset):

    def __init__(self, slide: openslide, patch_coors) -> None:
        super().__init__()
        self.slide = slide
        self.patch_coors = patch_coors
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.patch_coors)

    def __getitem__(self, idx: int):
        coor = self.patch_coors[idx]
        img = self.slide.read_region((coor[0], coor[1]), 0, (coor[2], coor[3])).convert('RGB')
        return self.transform(img)


def analysis_patches(slide_dir, patch_coors):
    slide = openslide.open_slide(slide_dir)

    dataset = TissueDataset(slide, patch_coors)
    test_dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    res = None

    for idx, image in enumerate(tqdm(test_dataloader)):
        image = image.to(device, dtype=torch.float)
        output = analysis_model(image)
        output = F.softmax(output, dim=1)  # add
        output = output * 10
        output = output.detach()

        try:
            res = torch.cat((res, output), dim=0)
        except:
            res = output

    return res


def generate_patch_extract_feature():
    tmp_file_root = 'tmpsvs'
    svs_dir = os.path.join(BASE_PATH, 'svs_directory')

    for root, dirs, files in os.walk(svs_dir):
        print(root)
        idx = 0
        random.shuffle(files)
        for file in files:
            idx += 1

            print(f'{idx} out of {len(files)}')

            file_path = os.path.join(root, file)

            feature_save_file_path = os.path.join(feature_save_path, f'{file[:-4]}_fts.npy')

            if os.access(feature_save_file_path, os.F_OK):
                continue

            print("extract ", file)

            tmp_file_path = os.path.join(tmp_file_root, file)
            try:
                os.system(f'cp "{file_path}" "{tmp_file_path}"')
            except:
                print("copy error")
                os.system(f'rm "{tmp_file_path}"')
                continue

            cfg = {
                'vis_folder': 'tmpvis',
                'alpha': 0.7,
                'frac': 16,
                'background_filter': True,
                'patch_size': 224,
                'patch_each_slide': 4000,
                'vis_patch_position': False,
                'vis_anno_mask': False
            }
            print(idx, " entering extract_patch")
            try:
                svs_patch = extract_patch(cfg, tmp_file_path)
            except:
                print("extract_patch error")
                os.system(f'rm "{tmp_file_path}"')
                print('sleep')
                time.sleep(1)
                continue

            print(idx, " out extract_patch")

            svs_patch_new = []
            for patch in svs_patch:
                patch = (patch[0], patch[1], cfg["patch_size"], cfg["patch_size"])
                svs_patch_new.append(patch)

            try:
                print(idx, " entering analysis_patch")
                patch_type = analysis_patches(tmp_file_path, svs_patch_new)
                patch_type_numpy = patch_type.cpu().numpy()
                print(idx, " out analysis_patch")

                print(idx, " entering extract_ft")
                svs_feature = extract_ft(tmp_file_path, svs_patch_new, depth=34, batch_size=256)
                svs_feature_numpy = svs_feature.cpu().numpy()
                print(idx, " out extract_ft")
            except:
                print("error")
                os.system(f'rm "{tmp_file_path}"')
                print('sleep')
                time.sleep(1)
                continue

            cat_feature = np.concatenate((svs_feature_numpy, patch_type_numpy), axis=1)

            np.save(feature_save_file_path, cat_feature)

            os.system(f'rm "{tmp_file_path}"')

            print('sleep')
            time.sleep(1)


if __name__ == '__main__':
    generate_patch_extract_feature()
