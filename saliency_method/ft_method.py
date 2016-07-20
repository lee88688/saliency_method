# coding: utf-8

import cv2
import numpy as np
from utils import normalize

__all__ = ['ft_saliency']

get_filter_kernel = lambda x, y: cv2.mulTransposed(cv2.getGaussianKernel(x, y), False)

def ft_saliency(img_lab):
    blur_img_lab = cv2.filter2D(img_lab, -1, get_filter_kernel(5, 5))
    blur_lm = blur_img_lab[:, :, 0].mean()
    blur_am = blur_img_lab[:, :, 1].mean()
    blur_bm = blur_img_lab[:, :, 2].mean()
    blur_sm = np.sqrt((blur_img_lab[:, :, 0] - blur_lm) ** 2 + (blur_img_lab[:, :, 1] - blur_am) ** 2 + (
        blur_img_lab[:, :, 2] - blur_bm) ** 2)

    return normalize(blur_sm)