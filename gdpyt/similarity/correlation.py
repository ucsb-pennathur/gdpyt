# import modules
from matplotlib import pyplot as plt
from scipy.signal import correlate2d
from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import curve_fit
from skimage.feature import match_template
import numpy as np

from gdpyt.subpixel_localization.gaussian import fit_2d_gaussian_on_ccorr

import logging

logger = logging.getLogger(__name__)


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


def barnkob_cross_correlation_equal_shape(img1, img2):
    if not (img1.shape == img2.shape):
        raise ValueError("Images must have the same shape for norm_function cross_correlation_equal_shape. Received shape {} and {}".format(img1.shape, img2.shape))
    img1 = np.nan_to_num(img1)
    img2 = np.nan_to_num(img2)
    return ((img1 - img1.mean()) * (img2 - img2.mean())).sum() / np.sqrt(np.power(img1 - img1.mean(), 2).sum() * np.power(img2 - img2.mean(), 2).sum())


def zero_norm_cross_correlation_equal_shape(img1, img2):
    if not (img1.shape == img2.shape):
        raise ValueError("Images must have the same shape for norm_function cross_correlation_equal_shape. Received shape {} and {}".format(img1.shape, img2.shape))
    img1 = np.nan_to_num(img1)
    img2 = np.nan_to_num(img2)
    return 1 / img1.size * correlate2d((img1 - img1.mean()) / img1.std(), (img2 - img2.mean()) / img2.std(), mode='valid', boundary='symm').item()


def sk_norm_cross_correlation(img1, img2):
    """
    skimage-based fast normalized cross correlation

    Steps:
        1. Determine which image is larger and deem this the 'image'; conversely, the smaller image is the template.
        2. Pass the appropriate image and template into the skimage-based template matching method.
        3. Get the peak correlation value and the x, y coordinates where the peak match was found.
    """
    if img1.size >= img2.size:
        result = match_template(img1, img2)

        """elif img2.shape[0] > img1.shape[0] and img2.shape[1] > img1.shape[1]:
        result = match_template(img2, img1)"""

    else:
        logger.warning("Unable to correlate mismatched templates: (img1, img2): ({}, {})".format(img1.shape, img2.shape))
        result = np.nan

    """# evaluate results
    res_length = np.floor(result.shape[0] / 2)

    # max correlation
    cm = np.max(result)

    # x,y coordinates in the image space where the highest correlation was found
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    x = x - res_length
    y = y - res_length

    # sub-pixel localization
    if np.size(result) > 5:
        xg, yg = fit_2d_gaussian_on_ccorr(result, x, y)
        if xg is not None and yg is not None:
            xg = xg - res_length
            yg = yg - res_length
    else:
        xg, yg = None, None
        
    return cm, x, y, xg, yg
    """

    return result



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


def max_barnkob_cross_correlation(img1, img2):
    # Index with highest normalized correlation
    ncorr = match_template(img1, img2)
    ij = np.unravel_index(np.argmax(ncorr), ncorr.shape)
    img1_crop = img1[ij[0]: ij[0] + img2.shape[0], ij[1]: ij[1] + img2.shape[1]]
    return barnkob_cross_correlation_equal_shape(img1_crop, img2)


def akima_interpolation(z_calib, sim, max_idx):
    # find index of maximum image correlation
    x_interp = z_calib
    y_interp = sim

    if len(z_calib) < 3 or len(sim) < 3:
        return z_calib, sim

    # determine the bounds of the fit
    lower_index = np.maximum(0, max_idx - 1)
    upper_index = np.minimum(max_idx + 1, len(z_calib) - 1)

    if lower_index >= len(z_calib) - 2:
        lower_index = len(z_calib) - 3
    if upper_index < 2:
        upper_index = 2

    # fit Akima cubic polynomial
    x_local = np.linspace(z_calib[lower_index], z_calib[upper_index], 50)
    sim_interp = Akima1DInterpolator(x_interp, y_interp)(x_local)

    return x_local, sim_interp


def parabolic_interpolation(z_calib, sim, max_idx):

    # if there are only two values, we cannot fit a three-point estimator
    if len(z_calib) < 3 or len(sim) < 3:
        return z_calib, sim

    # determine the bounds of the fit
    lower_index = np.maximum(0, max_idx - 1)
    upper_index = np.minimum(max_idx + 1, len(z_calib) - 1)

    if lower_index >= len(z_calib) - 2:
        lower_index = len(z_calib) - 3
    if upper_index < 2:
        upper_index = 2

    lower_bound = z_calib[lower_index]
    upper_bound = z_calib[upper_index]

    # fit parabola
    popt, pcov = curve_fit(parabola, z_calib[lower_index:upper_index+1], sim[lower_index:upper_index+1])

    # create interpolation space and get resulting parabolic curve
    x_local = np.linspace(lower_bound, upper_bound, 50)
    sim_interp = parabola(x_local, *popt)

    return x_local, sim_interp

def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c

def compare_interpolation_methods(z_calib, sim, max_idx, true_z):
    """
    A function that can be inserted into GdpytCalibrationStack.infer_z to measure the difference in accuracy when
    the particle true_z coordinate is known (or inferred; i.e. from a calibration stack)
    """

    # determine the bounds of the fit
    lower_bound = np.maximum(0, max_idx - 1)
    upper_bound = np.minimum(max_idx + 2, len(z_calib) - 1)

    if lower_bound == len(z_calib) - 2:
        lower_bound -= 1
    if upper_bound < 3:
        upper_bound = 3

    # correlation
    z_corr = z_calib[np.argmax(sim)]

    # Parabolic estimator
    x_local_p, sim_interp_p = parabolic_interpolation(z_calib, sim, max_idx)
    x_max_p = np.argmax(sim_interp_p)
    z_p = x_local_p[x_max_p]

    # Akima 1D interpolator
    x_local_Akima, sim_interp_Akima = akima_interpolation(z_calib, sim, max_idx)
    x_max_Akima = np.argmax(sim_interp_Akima)
    z_Akima = x_local_Akima[x_max_Akima]

    # calculate errors
    err_corr = np.round(np.abs(true_z - z_corr), 4)
    err_p = np.round(np.abs(true_z - z_p), 4)
    err_Akima = np.round(np.abs(true_z - z_Akima), 4)

    if err_p < err_Akima:
        best_interp = 'Parabolic'
        better_percent = np.round((err_Akima - err_p) / err_Akima, 3) * 100
        better_err = err_p
    elif err_Akima < err_p:
        best_interp = 'Akima'
        better_percent = np.round((err_p - err_Akima) / err_p, 3) * 100
        better_err = err_Akima
    else:
        best_interp = 'Tie'
        better_percent = 0
        better_err = 0

    saveid = '{}_z{}-best-by-{}-percent-with-error-{}'.format(best_interp, true_z, better_percent, better_err)

    fig, ax = plt.subplots()

    ax.scatter(true_z, 1, color='black', marker='*', label=r'$z_{true}$')

    ax.scatter(z_calib[lower_bound:upper_bound], sim[lower_bound:upper_bound], color='red',
               label=r'$c_{m}$='+'{}'.format(err_corr))

    ax.plot(x_local_p, sim_interp_p, color='tab:blue', label='Parabolic')
    ax.scatter(x_local_p[x_max_p], sim_interp_p[x_max_p], color='blue', marker='X',
               label=r'$z_{p}$='+'{}'.format(err_p))

    ax.plot(x_local_Akima, sim_interp_Akima, color='tab:purple', label='Akima')
    ax.scatter(x_local_Akima[x_max_Akima], sim_interp_Akima[x_max_Akima], color='darkviolet', marker='d',
               label=r'$z_{A}$='+'{}'.format(err_Akima))

    ax.set_title(saveid)

    ax.set_xlabel('z')
    ax.set_ylabel(r'$c_m$')
    ax.grid(alpha=0.25)
    ax.legend()

    plt.show()