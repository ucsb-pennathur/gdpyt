from scipy.signal import correlate2d
import numpy as np

def cross_correlation_equal_shape(img1, img2):
    if not (img1.shape == img2.shape):
        raise ValueError("Images must have the same shape for function cross_correlation_equal_shape")
    return correlate2d(img1, img2, mode='valid', boundary='symm').item()

def norm_cross_correlation_equal_shape(img1, img2):
    if not (img1.shape == img2.shape):
        raise ValueError("Images must have the same shape for norm_function cross_correlation_equal_shape")
    return 1 / img1.size * correlate2d(img1 / img1.std(), img2 / img2.std(), mode='valid', boundary='symm').item()

def zero_norm_cross_correlation_equal_shape(img1, img2):
    if not (img1.shape == img2.shape):
        raise ValueError("Images must have the same shape for norm_function cross_correlation_equal_shape")
    return 1 / img1.size * correlate2d((img1 - img1.mean()) / img1.std(), (img2 - img2.mean()) / img2.std(), mode='valid', boundary='symm').item()
