# coding: utf-8

import os
from skimage import io
from saliency_method.sf_method import SF_Method
from saliency_method.ft_method import ft_saliency
from saliency_method.utils import *


if __name__ == '__main__':
    img = io.imread("(744).jpg")

    sf = SF_Method(img)
    sf.generate_features()
    io.imsave("uniqueness.png", feature_to_image(sf.uniqueness, sf.segments))
    io.imsave("distribution.png", feature_to_image(sf.distribution, sf.segments))
    sf_saliency = sf.saliency_assignment(3)
    io.imsave("sf_method.png", feature_to_image(sf_saliency, sf.segments))
    io.imsave("sf_method_up_sampling.png", up_sample(sf.img_lab, sf_saliency, sf.img_segments_mean, sf.coordinate_segments_mean))

    io.imsave("ft_method.png", ft_saliency(sf.img_lab))