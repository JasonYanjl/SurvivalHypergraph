import json
import os
import os.path as osp
from glob import glob
from itertools import product
from random import shuffle

import sys
install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)
from base_utils import BASE_PATH

import numpy as np
import openslide
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, star, opening
from tqdm import tqdm

BACKGROUND = 0
FOREGROUND = 255
SAMPLED = 2

SAMPLED_COLOR = [255, 0, 0]

SAMPLED_COLOR_LIGHT = [[255,0,0],[255,165,0],[0,0,255],[64,224,208],[127,255,0],
                       [255,255,0],[0,191,255],[255,99,250],[138,150,226],[120,0,255]]

# SAMPLED_COLOR_LIGHT = [[255, 0, 0], [255, 146, 76]]
# SAMPLED_COLOR_LIGHT = [[54, 98, 164], [107, 189, 167]]


def generate_patch(cfg):
    regenerate_patch = cfg['regenerate_patch']
    coor_folder = cfg['coor_folder']

    # get file names
    pos_slide_dir = sorted(glob(osp.join(cfg['tumor_data_folder'], '*.svs')))
    # pos_anno_dir = sorted(glob(osp.join(cfg['tumor_anno_folder'], '*.xml')))
    neg_slide_dir = sorted(glob(osp.join(cfg['normal_data_folder'], '*.svs')))

    # check slides and annotation, and print slide information
    all_slide_dir = list()
    # for negative slides
    for slide_dir in neg_slide_dir:
        slide_name: str = osp.basename(slide_dir).split('.')[0]
        all_slide_dir.append((slide_name, slide_dir, None))
    # for positive slides
    for slide_dir in pos_slide_dir:
        slide_name: str = osp.basename(slide_dir).replace('.svs', '')
        # anno_dir = osp.join(cfg['tumor_anno_folder'], slide_name + '.xml')
        # assert anno_dir in pos_anno_dir, f'cann\'t find the corresponding annotation file of {slide_name}'
        # all_slide_dir.append((slide_name, slide_dir, anno_dir))
        all_slide_dir.append((slide_name, slide_dir))
    # summarize information
    pos_size, neg_size = len(pos_slide_dir), len(neg_slide_dir)
    print(f'find {pos_size} tumor slides, {neg_size} normal slides!')

    # process each tumor slides
    print('sampling positive patches!')
    # for slide_name, slide_dir, anno_dir in tqdm(all_slide_dir):
    for slide_name, slide_dir in tqdm(all_slide_dir):
        coor_file = osp.join(coor_folder, slide_name + '.json')
        if osp.exists(coor_file) and not regenerate_patch:
            continue
        print(f'extracting patches from {slide_dir}')
        # patch_coor = extract_patch(cfg, slide_dir, anno_dir)
        patch_coor = extract_patch(cfg, slide_dir)
        with open(coor_file, 'w') as f:
            f.write(json.dumps(patch_coor) + '\n')


# def extract_patch(cfg, slide_dir, anno_dir):
def extract_patch(cfg, slide_dir):
    slide = openslide.open_slide(slide_dir)
    slide_name = osp.basename(slide_dir).replace('.svs', '')
    frac = cfg['frac']  # 32
    background_filter = cfg['background_filter']
    patch_size = cfg['patch_size']
    mini_patch_size = patch_size // frac
    patch_each_slide = cfg['patch_each_slide']
    mask_size = np.ceil(np.array(slide.level_dimensions[0]) / frac).astype(np.int)
    mask_level = get_level(slide, mask_size)
    mask = np.zeros((mask_size[1], mask_size[0]), np.uint8)

    th_mask = generate_img_bg_mask(slide, mask_level, mask_size)

    assert (mask_size[1], mask_size[0]) == th_mask.shape

    # generate the available mask
    # if anno_dir is not None:
    #     anno_mask = generate_mask(anno_dir, mask_level, mask_size)
    #     mask[th_mask != 0] = anno_mask[th_mask != 0]
    # else:
    #     mask[th_mask != 0] = 1
    mask[th_mask != 0] = 1
    # extract patches from available area
    patches = []
    num_row, num_col = mask.shape
    num_row = num_row - mini_patch_size
    num_col = num_col - mini_patch_size

    row_col = list(product(range(num_row), range(num_col)))
    shuffle(row_col)
    cnt = 0

    # attention center
    H_min = int(np.ceil(mini_patch_size / 8))
    H_max = int(np.ceil(mini_patch_size / 8 * 7))
    W_min = int(np.ceil(mini_patch_size / 8))
    W_max = int(np.ceil(mini_patch_size / 8 * 7))
    # half of the center
    th_num = int(np.ceil((mini_patch_size * 3 / 4 * mini_patch_size * 3 / 4)))

    for row, col in row_col:
        if cnt >= patch_each_slide:
            break
        mask_patch = mask[row:row + mini_patch_size, col: col + mini_patch_size]
        origin = (int(col * frac), int(row * frac))
        if np.count_nonzero(mask_patch[H_min:H_max, W_min:W_max]) >= th_num:
            # filter those white background for normal slides
            # if anno_dir is None and background_filter:
            if background_filter:
                if is_bg(slide, origin, patch_size):
                    continue
            patches.append(origin)
            cnt += 1

    if cfg['vis_patch_position']:
        vis_ov_mask_img(cfg, slide, slide_name, mask_size, patches, patch_size)
    if cfg['vis_anno_mask']:
        vis_anno_mask_img(cfg, slide, slide_name, mask_size, mask)

    return patches


# get the just size that equal to mask_size
def get_level(slide: openslide, size):
    level = slide.level_count - 1
    while level >= 0 and slide.level_dimensions[level][0] < size[0] and \
            slide.level_dimensions[level][1] < size[1]:
        level -= 1
    return level


def generate_img_bg_mask(slide: openslide, mask_level, mask_size, get_img=False, train=True):
    th_img = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level])
    th_img = th_img.resize(mask_size)
    th_mask = threshold_downsample_level(th_img, train)
    # th_mask = seg_dfs(th_img)

    if get_img:
        return th_img, th_mask
    else:
        th_img.close()
        return th_mask


# background segmentation algorithm 1
def threshold_downsample_level(img, train=True):
    # calculate the overview level size and retrieve the image
    img_hsv = img.convert('HSV')
    img_hsv_np = np.array(img_hsv)

    # dilate image and then threshold the image
    schannel = img_hsv_np[:, :, 1]
    mask = np.zeros(schannel.shape)
    if train:
        schannel = dilation(schannel, star(7))
    else:
        schannel = dilation(schannel, star(3))
    schannel = ndimage.gaussian_filter(schannel, sigma=(5, 5), order=0)
    threshold_global = threshold_otsu(schannel)

    mask[schannel > threshold_global] = FOREGROUND
    mask[schannel <= threshold_global] = BACKGROUND

    import scipy.misc   # check the result
    # scipy.misc.imsave('outfile.jpg', schannel)

    return mask


# background segmentation algorithm 2
def seg_dfs(img):
    img_np = np.asarray(img)
    img_np_g = img_np[:, :, 1]
    shape = img_np_g.shape
    mask = np.ones(shape).astype(np.uint8) * FOREGROUND
    searched = np.zeros((shape)).astype(np.bool)
    coor = []
    init_val = 0

    def inRange(val):
        return init_val - 10 <= val <= init_val + 10

    def addSeed_initVal():
        val1 = img_np_g[:, 0].mean()
        val2 = img_np_g[0, :].mean()
        val3 = img_np_g[:, shape[1] - 1].mean()
        val4 = img_np_g[shape[0] - 1, 0].mean()
        val = np.max((val1, val2, val3, val4))
        for idx in range(shape[0]):
            # L
            coor.append({'x': idx, 'y': 0})
            searched[idx, 0] = True
            # R
            coor.append({'x': idx, 'y': shape[1] - 1})
            searched[idx, shape[1] - 1] = True
        for idx in range(shape[1]):
            # U
            coor.append({'x': 0, 'y': idx})
            searched[0, idx] = True
            # D
            coor.append({'x': shape[0] - 1, 'y': idx})
            searched[shape[0] - 1, idx] = True
        return val

    def isPixel(x, y):
        return (0 <= x < shape[0]) and (0 <= y < shape[1])

    def deal(x, y):
        if isPixel(x, y) and not searched[x, y] and inRange(img_np_g[x, y]):
            coor.append({'x': x, 'y': y})
            searched[x, y] = True
            mask[x, y] = BACKGROUND

    init_val = addSeed_initVal()
    # print('init val: %d' % init_val)

    while coor:
        x = coor[0]['x']
        y = coor[0]['y']
        if x == 0 or y == 0 \
                or x == shape[0] - 1 or y == shape[1] - 1:
            deal(x, y)
        del coor[0]
        deal(x + 1, y)
        deal(x, y + 1)
        deal(x - 1, y)
        deal(x, y - 1)

    mask = opening(mask, star(5))
    # mask = erosion(mask, star(3))
    mask = dilation(mask, star(3))
    return mask


def generate_mask(anno_dir: str, mask_level, mask_size):
    tif_mask = anno_dir.replace('.xml', '.tif')
    mask = openslide.open_slide(tif_mask)
    mask = mask.read_region((0, 0), mask_level, mask.level_dimensions[mask_level]).convert('L')
    mask = mask.resize(mask_size)
    mask = np.asarray(mask).astype(np.uint8)
    return mask


def is_bg(slide, origin, patch_size):
    img = slide.read_region(origin, 0, (patch_size, patch_size))
    # bad case is background
    if np.array(img)[:, :, 1].mean() > 200:  # is bg
        img.close()
        return True
    else:
        img.close()
        return False


def get_sampled_patch_mask(slide: openslide, coors, patch_size, mask_size):
    level = get_level(slide, (40000, 40000)) + 1
    size = slide.level_dimensions[level]
    sampled_mask = np.zeros((size[1], size[0]), np.uint8)
    frac = size[0] * 1.0 / slide.level_dimensions[0][0]
    mini_patch_size = int(patch_size * frac)
    for coor in coors:
        mini_coor = (int(coor[0] * frac), int(coor[1] * frac))
        sampled_mask[mini_coor[1]:mini_coor[1] + mini_patch_size, mini_coor[0]:mini_coor[0] + mini_patch_size] = SAMPLED
    sampled_mask = np.asarray(Image.fromarray(sampled_mask).resize(mask_size))
    return sampled_mask


def get_sampled_patch_mask_channels(slide: openslide, coors, patch_size, mask_size, enlarge_scale=1):
    level = get_level(slide, (40000, 40000)) + 1
    size = slide.level_dimensions[level]
    sampled_mask = np.zeros((size[1], size[0]), np.uint8)
    frac = size[0] * 1.0 / slide.level_dimensions[0][0]
    mini_patch_size = int(patch_size * frac)
    for coor in coors:
        mini_coor = (int(coor[0] * frac), int(coor[1] * frac))
        if enlarge_scale != 1:
            enlarge_high = (enlarge_scale - enlarge_scale // 2) * mini_patch_size
            enlarge_low = enlarge_scale // 2 * mini_patch_size
            sampled_mask[max(0,mini_coor[1]-enlarge_low):min(mini_coor[1] + enlarge_high, sampled_mask.shape[0]), max(0, mini_coor[0]-enlarge_low):min(mini_coor[0] + enlarge_high, sampled_mask.shape[1])] = SAMPLED
        else:
            sampled_mask[mini_coor[1]:mini_coor[1] + mini_patch_size, mini_coor[0]:mini_coor[0] + mini_patch_size] = SAMPLED
    sampled_mask = np.asarray(Image.fromarray(sampled_mask).resize(mask_size))
    return sampled_mask


def fusion_mask_img(img, mask_np, alpha):
    img_np = np.asarray(img)
    assert (img.size[1], img.size[0]) == mask_np.shape
    img_mask = img_np.copy()

    # if (mask_np == SAMPLED).any():
    #     img_mask[mask_np == SAMPLED] = alpha * img_np[mask_np == SAMPLED] + \
    #                                    (1 - alpha) * np.array(SAMPLED_COLOR)
    if (mask_np != 0).any():
        img_mask[mask_np != 0] = alpha * img_np[mask_np != 0] + \
                                 (1 - alpha) * np.array(SAMPLED_COLOR)
    return Image.fromarray(img_mask)


def fusion_mask_img_channels(img, mask_np, alpha, channel):
    img_np = np.asarray(img)
    assert (img.size[1], img.size[0]) == mask_np.shape
    img_mask = img_np.copy()

    # if (mask_np == SAMPLED).any():
    #     img_mask[mask_np == SAMPLED] = alpha * img_np[mask_np == SAMPLED] + \
    #                                    (1 - alpha) * np.array(SAMPLED_COLOR)
    if (mask_np != 0).any():
        img_mask[mask_np != 0] = alpha * img_np[mask_np != 0] + \
                                 (1 - alpha) * np.array(SAMPLED_COLOR_LIGHT[channel])
    return Image.fromarray(img_mask)


def vis_origin(cfg, slide, slide_name, patch_size, location):
    mask_size = np.ceil(np.array(slide.level_dimensions[0]) / cfg['frac']).astype(np.int)
    # mask_level = get_level(slide, mask_size)
    level = get_level(slide, (40000, 40000)) + 1
    raw_img = slide.read_region(location, 0,
                                (patch_size, patch_size)).convert('RGB')
    raw_img.save(osp.join(cfg['vis_folder'], slide_name + '.jpg'))
    raw_img.close()


def vis_ov_mask_img(cfg, slide: openslide, slide_name, mask_size, coors, patch_size):
    mask_np = get_sampled_patch_mask(slide, coors, patch_size, mask_size)
    save_name = slide_name + '_sampled.jpg'
    vis_mask_img(cfg['vis_folder'], slide, mask_np, mask_size, save_name, cfg['alpha'])


def vis_anno_mask_img(cfg, slide: openslide, slide_name, mask_size, mask_np):
    save_name = slide_name + '_mask.jpg'
    vis_mask_img(cfg['vis_folder'], slide, mask_np, mask_size, save_name, cfg['alpha'])


def vis_ov_mask_img_channels(cfg, slide: openslide, slide_name, mask_size, coors, patch_size, is_time_low = True, color_indices = list(range(len(SAMPLED_COLOR_LIGHT))), enlarge_scale=1):
    mask_nps = []
    for _coor in coors:
        mask_np = get_sampled_patch_mask_channels(slide, _coor, patch_size, mask_size, enlarge_scale=enlarge_scale)
        mask_nps.append(mask_np)
    save_name = slide_name + '_sampled.jpg'
    vis_mask_img_channels(cfg['vis_folder'], slide, mask_nps, mask_size, save_name, cfg['alpha'], color_indices, is_time_low=is_time_low)


def vis_mask_img_channels(save_dir, slide: openslide, mask_nps, mask_size, save_name, alpha, color_indices, is_time_low=True):
    mask_level = get_level(slide, mask_size)
    raw_img = slide.read_region((0, 0), mask_level,
                                slide.level_dimensions[mask_level]).convert('RGB')
    raw_img = raw_img.resize(mask_size)

    sampled_patch_img = raw_img

    if is_time_low:
        for i in range(len(mask_nps)-1, -1, -1):
            mask_np = mask_nps[i]
            sampled_patch_img = fusion_mask_img_channels(sampled_patch_img, mask_np, alpha, channel=color_indices[i])
    else:
        for i in range(0, len(mask_nps), 1):
            mask_np = mask_nps[i]
            sampled_patch_img = fusion_mask_img_channels(sampled_patch_img, mask_np, alpha, channel=color_indices[i])

    sampled_patch_img.save(osp.join(save_dir, save_name))
    sampled_patch_img.close()

    raw_img.close()


def vis_mask_img(save_dir, slide: openslide, mask_np, mask_size, save_name, alpha):
    mask_level = get_level(slide, mask_size)
    raw_img = slide.read_region((0, 0), mask_level,
                                slide.level_dimensions[mask_level]).convert('RGB')
    raw_img = raw_img.resize(mask_size)

    sampled_patch_img = fusion_mask_img(raw_img, mask_np, alpha)
    sampled_patch_img.save(osp.join(save_dir, save_name))
    sampled_patch_img.close()
    raw_img.close()

