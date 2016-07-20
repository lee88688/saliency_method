# coding: utf-8

import numpy as np
from utils import normalize
from skimage.color import rgb2lab
from skimage.segmentation import slic
from scipy.spatial.distance import cdist

__all__ = ['SF_Method']

class SF_Method:

    def __init__(self, img_array, sigma_uniqueness=50, sigma_distribution=20):
        self._sigma_uniqueness = sigma_uniqueness
        self._sigma_distribution = sigma_distribution
        self._img = img_array

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, new_img):
        self._img = new_img

    @property
    def sigma_uniqueness(self):
        return self._sigma_uniqueness

    @sigma_uniqueness.setter
    def sigma_uniqueness(self, sigma):
        self._sigma_uniqueness = sigma

    @property
    def sigma_distribution(self):
        return self._sigma_distribution

    @sigma_distribution.setter
    def sigma_distribution(self, sigma):
        self._sigma_distribution = sigma

    def generate_features(self):
        # prepare variables
        img_lab = rgb2lab(self._img)
        segments = slic(img_lab, n_segments=500, compactness=30.0, convert2lab=False)
        max_segments = segments.max() + 1

        # create x,y feather
        shape = self._img.shape
        a = shape[0]
        b = shape[1]
        x_axis = np.linspace(0, b - 1, num=b)
        y_axis = np.linspace(0, a - 1, num=a)

        x_coordinate = np.tile(x_axis, (a, 1,))  # 创建X轴的坐标表
        y_coordinate = np.tile(y_axis, (b, 1,))  # 创建y轴的坐标表
        y_coordinate = np.transpose(y_coordinate)

        coordinate_segments_mean = np.zeros((max_segments, 2))

        # create lab feather
        img_l = img_lab[:, :, 0]
        img_a = img_lab[:, :, 1]
        img_b = img_lab[:, :, 2]

        img_segments_mean = np.zeros((max_segments, 3))

        for i in xrange(max_segments):
            segments_i = segments == i

            coordinate_segments_mean[i, 0] = x_coordinate[segments_i].mean()
            coordinate_segments_mean[i, 1] = y_coordinate[segments_i].mean()

            img_segments_mean[i, 0] = img_l[segments_i].mean()
            img_segments_mean[i, 1] = img_a[segments_i].mean()
            img_segments_mean[i, 2] = img_b[segments_i].mean()

        # element distribution
        wc_ij = np.exp(-cdist(img_segments_mean, img_segments_mean) ** 2 / (2 * self._sigma_distribution ** 2))
        wc_ij = wc_ij / wc_ij.sum(axis=1)[:, None]
        mu_i = np.dot(wc_ij, coordinate_segments_mean)
        distribution = np.dot(wc_ij, np.linalg.norm(coordinate_segments_mean - mu_i, axis=1) ** 2)
        distribution = normalize(distribution)
        distribution = np.array([distribution]).T

        # element uniqueness feature
        wp_ij = np.exp(
            -cdist(coordinate_segments_mean, coordinate_segments_mean) ** 2 / (2 * self._sigma_uniqueness ** 2))
        wp_ij = wp_ij / wp_ij.sum(axis=1)[:, None]
        uniqueness = np.sum(cdist(img_segments_mean, img_segments_mean) ** 2 * wp_ij, axis=1)
        uniqueness = normalize(uniqueness)
        uniqueness = np.array([uniqueness]).T

        # save features and variables
        self.img_lab = img_lab
        self.segments = segments
        self.img_segments_mean = img_segments_mean
        self.coordinate_segments_mean = coordinate_segments_mean
        self.uniqueness = uniqueness
        self.distribution = distribution

    def saliency_assignment(self, k):
        return self.uniqueness * np.exp(-k*self.distribution)