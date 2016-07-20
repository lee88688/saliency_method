# coding: utf-8

import numpy as np
from scipy.spatial.distance import cdist

__all__ = ['normalize', 'feature_to_image', 'up_sample']

normalize = lambda s: (s - s.min()) / (s.max() - s.min())

def feature_to_image(feature, segments):
    '''
    translate saliency feature into its segments images,
    the same sagment region is assigned to the same value.
    :param feature: saliency feature
    :param segments: segment labels
    :return:
    '''
    max_segments = segments.max() + 1
    if max_segments != feature.size:
        raise NameError("feature size does not match segments.")

    feature_img = np.zeros_like(segments, dtype=np.float64)

    for i in xrange(max_segments):
        segments_i = segments == i
        feature_img[segments_i] = feature[i]

    return feature_img


def up_sample(img_lab, saliency, img_segments_mean, coordinate_segments_mean):
    '''
    use up-sampling method that is used in SF method to generate full revolution
    of saliency map.
    :param img_lab: image matrix in lab color space
    :param saliency: saliency value of superpixels
    :param img_segments_mean: mean values of color in each superpixel in lab color space
    :param coordinate_segments_mean: mean values of position in each superpixel
    :return:
    '''
    size = img_lab.size / 3
    shape = img_lab.shape
    # size = img.size
    a = shape[0]
    b = shape[1]
    x_axis = np.linspace(0, b - 1, num=b)
    y_axis = np.linspace(0, a - 1, num=a)

    x_coordinate = np.tile(x_axis, (a, 1,))  # create x coordinate
    y_coordinate = np.tile(y_axis, (b, 1,))  # create y coordinate
    y_coordinate = np.transpose(y_coordinate)

    c_i = np.concatenate(
        (img_lab[:, :, 0].reshape(size, 1), img_lab[:, :, 1].reshape(size, 1), img_lab[:, :, 2].reshape(size, 1)),
        axis=1)
    p_i = np.concatenate((x_coordinate.reshape(size, 1), y_coordinate.reshape(size, 1)), axis=1)
    w_ij = np.exp(
        -1.0 / (2 * 30) * (cdist(c_i, img_segments_mean) ** 2 + cdist(p_i, coordinate_segments_mean) ** 2))
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]
    if len(saliency.shape) != 2 or saliency.shape[1] != 1:
        saliency = saliency[:, None]
    saliency_pixel = np.dot(w_ij, saliency)
    return saliency_pixel.reshape(shape[0:2])