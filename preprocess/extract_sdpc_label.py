import os
import sys
import random
from PIL import Image

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

import time, json

from paddleocr import PaddleOCR, draw_ocr

def is_chinese(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def imgrotate(png_file):
    im = Image.open(png_file)
    ng = im.transpose(Image.ROTATE_270)
    ng.save(png_file)

def getsdpclabel(png_file):
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
    res = []
    for _shift in range(4):
        result = ocr.ocr(png_file, cls=True)
        for i in range(len(result)):
            if is_chinese(result[i][1][0]):
                continue
            if result[i][1][1] < 0.85:
                continue
            if result[i][1][0] not in res:
                res.append(result[i][1][0])
        imgrotate(png_file)

    res.sort(key=lambda i: len(i), reverse=True)
    return res


