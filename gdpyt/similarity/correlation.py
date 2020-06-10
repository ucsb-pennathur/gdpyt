from scipy.signal import correlate2d
from scipy.interpolate import Akima1DInterpolator
from skimage.feature import match_template
import numpy as np


def cross_correlation_equal_shape(img1, img2):
    if not (img1.shape == img2.shape):
        raise ValueError("Images must have the same shape for function cross_correlation_equal_shape. Received shape {} and {}".format(img1.shape, img2.shape))
    img1 = np.nan_to_num(img1)
    img2 = np.nan_to_num(img2)
    return correlate2d(img1, img2, mode='valid', boundary='symm').item()


def norm_cross_correlation_equal_shape(img1, img2):
    if not (img1.shape == img2.shape):
        raise ValueError("Images must have the same shape for norm_function cross_correlation_equal_shape. Received shape {} and {}".format(img1.shape, img2.shape))
    img1 = np.nan_to_num(img1)
    img2 = np.nan_to_num(img2)
    return 1 / img1.size * correlate2d(img1 / img1.std(), img2 / img2.std(), mode='valid', boundary='symm').item()


def zero_norm_cross_correlation_equal_shape(img1, img2):
    if not (img1.shape == img2.shape):
        raise ValueError("Images must have the same shape for norm_function cross_correlation_equal_shape. Received shape {} and {}".format(img1.shape, img2.shape))
    img1 = np.nan_to_num(img1)
    img2 = np.nan_to_num(img2)
    return 1 / img1.size * correlate2d((img1 - img1.mean()) / img1.std(), (img2 - img2.mean()) / img2.std(), mode='valid', boundary='symm').item()


def max_cross_correlation(img1, img2):
    # Index with highest normalized correlation
    ncorr = match_template(img1, img2)
    ij = np.unravel_index(np.argmax(ncorr), ncorr.shape)
    img1_crop = img1[ij[0]: ij[0] + img2.shape[0], ij[1]: ij[1] + img2.shape[1]]
    return cross_correlation_equal_shape(img1_crop, img2)


def max_norm_cross_correlation(img1, img2):
    # Index with highest normalized correlation
    ncorr = match_template(img1, img2)
    ij = np.unravel_index(np.argmax(ncorr), ncorr.shape)
    img1_crop = img1[ij[0]: ij[0] + img2.shape[0], ij[1]: ij[1] + img2.shape[1]]
    return norm_cross_correlation_equal_shape(img1_crop, img2)


def max_zero_norm_cross_correlation(img1, img2):
    # Index with highest normalized correlation
    ncorr = match_template(img1, img2)
    ij = np.unravel_index(np.argmax(ncorr), ncorr.shape)
    img1_crop = img1[ij[0]: ij[0] + img2.shape[0], ij[1]: ij[1] + img2.shape[1]]
    return zero_norm_cross_correlation_equal_shape(img1_crop, img2)


def akima_interpolation(z_calib, sim, max_idx):
    # find index of maximum image correlation
    x_interp = z_calib
    y_interp = sim

    if len(z_calib) < 3 or len(sim) < 3:
        return z_calib, sim

    # fit Akima cubic polynomial
    x_local = np.linspace(z_calib[np.maximum(0, max_idx - 1)],
                          z_calib[np.minimum(max_idx + 1, len(z_calib) - 1)], 50)    # create linear space for fitting function
    sim_interp = Akima1DInterpolator(x_interp, y_interp)(x_local)               #

    return x_local, sim_interp

