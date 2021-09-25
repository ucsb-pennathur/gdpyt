"""
Local implementation of TrackPy's sub-pixel localizatin method.

Copyright Notice and Statement for the trackpy Project
===================================================

   Copyright (c) 2013-2014 trackpy contributors
   https://github.com/soft-matter/trackpy
   All rights reserved

"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.ndimage.filters import uniform_filter1d, correlate1d
from scipy.ndimage.fourier import fourier_gaussian

import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def refine_coords_via_centroid(raw_image, image, radius, coords, max_iterations=10, shift_thresh=0.6, characterize=False,
                               show_plot=False):
    """
    Refine coordinates based on Crocker-Grier centroid method:
    Crocker, J.C., Grier, D.G. https://dx.doi.org/10.1006/jcis.1996.0217

    Parameters
    ----------
    raw_image: full-size raw image.
    image: processed full-size image after applying ellipsoid mask which makes all background pixel values = 0.
    radius: tuple of integers
    coords:
    max_iterations: must be greater than 0. Recommended is 10
    shift_thresh
    characterize

    Returns
    -------

    """

    ndim = image.ndim
    isotropic = np.all(radius[1:] == radius[:-1])
    mask = binary_mask(radius, ndim).astype(np.uint16) # creates an ellipsoid mask of 1's and 0's.

    # Declare arrays that we will fill iteratively through loop.
    N = coords.shape[0]
    final_coords = np.empty_like(coords, dtype=np.float64)
    mass = np.empty(N, dtype=np.float64)
    raw_mass = np.empty(N, dtype=np.float64)
    if characterize:
        if isotropic:
            Rg = np.empty(N, dtype=np.float64)
        else:
            Rg = np.empty((N, len(radius)), dtype=np.float64)
        ecc = np.empty(N, dtype=np.float64)
        signal = np.empty(N, dtype=np.float64)

    ogrid = np.ogrid[[slice(0, i) for i in mask.shape]]  # for center of mass
    ogrid = [g.astype(float) for g in ogrid]

    for feat, coord in enumerate(coords):
        for iteration in range(max_iterations):
            # Define the circular neighborhood of (x, y).
            rect = tuple([slice(c - r, c + r + 1) for c, r in zip(coord, radius)]) # gets the particle image template.
            neighborhood = mask * image[rect] # applies the mask to the template to remove backgroud pixel values.
            cm_n = _center_of_mass(neighborhood, radius, ogrid)
            cm_i = cm_n - radius + coord  # image coords

            off_center = cm_n - radius
            logger.debug('off_center: %f', off_center)
            if np.all(np.abs(off_center) < shift_thresh):
                break  # Accurate enough.
            # If we're off by more than half a pixel in any direction, move..
            coord[off_center > shift_thresh] += 1
            coord[off_center < -shift_thresh] -= 1
            # Don't move outside the image!
            upper_bound = np.array(image.shape) - 1 - radius
            coord = np.clip(coord, radius, upper_bound).astype(int)

        # stick to yx column order
        final_coords[feat] = cm_i

        if show_plot:
            fig, ax = plt.subplots(ncols=2)
            ax[0].imshow(neighborhood)
            ax[0].scatter(cm_n[1], cm_n[0], marker='*', color='red')
            ax[0].set_title('Neighborhood')
            ax[1].imshow(image)
            ax[1].set_title('Image')
            plt.show()

        # Characterize the neighborhood of our final centroid.
        mass[feat] = neighborhood.sum()
        if characterize:
            logger.warning("Characterization not yet implemented")
            characterize = False

    return np.column_stack([final_coords, mass])

def _center_of_mass(x, radius, grids):
    """
    x: the masked image where all pixel values outside ellipsoid mask are = 0.
    radius: the estimated particle image radius.
    grids: integer sequence (i.e. 0, 1, 2, ..., M/N) where M/N is the size of the (M x N) image template respectively.
    """
    normalizer = x.sum() # the sum of all pixel values in the image template.
    if normalizer == 0:  # avoid divide-by-zero errors
        return np.array(radius)
    center_of_mass = np.array([(x * grids[dim]).sum() / normalizer for dim in range(x.ndim)])
    return center_of_mass

def binary_mask(radius, ndim):
    "Elliptical mask in a rectangular array"
    radius = validate_tuple(radius, ndim)
    points = [np.arange(-rad, rad + 1) for rad in radius]
    if len(radius) > 1:
        coords = np.array(np.meshgrid(*points, indexing="ij"))
    else:
        coords = np.array([points[0]])
    r = [(coord/rad)**2 for (coord, rad) in zip(coords, radius)]
    return sum(r) <= 1

def validate_tuple(value, ndim):
    if not hasattr(value, '__iter__'):
        return (value,) * ndim
    if len(value) == ndim:
        return tuple(value)
    raise ValueError("List length should have same length as image dimensions.")


"""
The below is a copy of the bandpass filtering method implemented in trackpy. 
The filter returns the image array with all background pixels = 0.
"""
def bandpass(image, lshort, llong, threshold=None, truncate=4):
    """Remove noise and background variation.

    Convolve with a Gaussian to remove short-wavelength noise and subtract out
    long-wavelength variations by subtracting a running average. This retains
    features of intermediate scale.

    The lowpass implementation relies on scipy.ndimage.filters.gaussian_filter,
    and it is the fastest way known to the authors of performing a bandpass in
    Python.

    Parameters
    ----------
    image : ndarray
    lshort : number or tuple
        Size of the gaussian kernel with which the image is convolved.
        Provide a tuple for different sizes per dimension.
    llong : integer or tuple
        The size of rolling average (square or rectangular kernel) filter.
        Should be odd and larger than the particle diameter.
        When llong <= lshort, an error is raised.
        Provide a tuple for different sizes per dimension.
    threshold : float or integer
        Clip bandpass result below this value. Thresholding is done on the
        already background-subtracted image.
        By default, 1 for integer images and 1/255 for float images.
    truncate : number, optional
        Determines the truncation size of the gaussian kernel. Default 4.

    Returns
    -------
    result : array
        the bandpassed image

    See Also
    --------
    lowpass, boxcar, legacy_bandpass, legacy_bandpass_fftw

    Notes
    -----
    The boxcar size and shape changed in v0.4: before, the boxcar had a
    circular kernel with radius `llong`, now it is has a square kernel that
    has an edge length of `llong` (twice as small!).
    """
    lshort = validate_tuple(lshort, image.ndim)
    llong = validate_tuple(llong, image.ndim)
    if np.any([x >= y for (x, y) in zip(lshort, llong)]):
        raise ValueError("The smoothing length scale must be larger than " +
                         "the noise length scale.")
    if threshold is None:
        if np.issubdtype(image.dtype, np.integer):
            threshold = 1
        else:
            threshold = 1/2**14. # note: the original was "255."
    background = boxcar(image, llong)
    result = lowpass(image, lshort, truncate)
    result -= background
    return np.where(result >= threshold, result, 0)

def boxcar(image, size):
    """Compute a rolling (boxcar) average of an image.

    The kernel is square or rectangular.

    Parameters
    ----------
    image : ndarray
    size : number or tuple
        Size of rolling average (square or rectangular kernel) filter. Should
        be odd and larger than the particle diameter.
        Provide a tuple for different sizes per dimension.

    Returns
    -------
    result : array
        the rolling average image

    See Also
    --------
    bandpass
    """
    size = validate_tuple(size, image.ndim)
    if not np.all([x & 1 for x in size]):
        raise ValueError("Smoothing size must be an odd integer. Round up.")
    result = image.copy()
    for axis, _size in enumerate(size):
        if _size > 1:
            uniform_filter1d(result, _size, axis, output=result,
                             mode='nearest', cval=0)
    return result

def lowpass(image, sigma=1, truncate=4):
    """Remove noise by convolving with a Gaussian.

    Convolve with a Gaussian to remove short-wavelength noise.

    The lowpass implementation relies on scipy.ndimage.filters.gaussian_filter,
    and it is the fastest way known to the authors of performing a bandpass in
    Python.

    Parameters
    ----------
    image : ndarray
    sigma : number or tuple, optional
        Size of the gaussian kernel with which the image is convolved.
        Provide a tuple for different sizes per dimension. Default 1.
    truncate : number, optional
        Determines the truncation size of the convolution kernel. Default 4.

    Returns
    -------
    result : array
        the processed image, as float

    See Also
    --------
    bandpass
    """
    sigma = validate_tuple(sigma, image.ndim)
    result = np.array(image, dtype=float)
    for axis, _sigma in enumerate(sigma):
        if _sigma > 0:
            correlate1d(result, gaussian_kernel(_sigma, truncate), axis,
                        output=result, mode='constant', cval=0.0)
    return result

def gaussian_kernel(sigma, truncate=4.0):
    "1D discretized gaussian"
    lw = int(truncate * sigma + 0.5)
    x = np.arange(-lw, lw+1)
    result = np.exp(x**2/(-2*sigma**2))
    return result / np.sum(result)

def grey_dilation(image, separation, percentile=64, margin=None, precise=True):
    """Find local maxima whose brightness is above a given percentile.

    Parameters
    ----------
    image : ndarray
        For best performance, provide an integer-type array. If the type is not
        of integer-type, the image will be normalized and coerced to uint8.
    separation : number or tuple of numbers
        Minimum separation between maxima. See precise for more information.
    percentile : float in range of [0,100], optional
        Features must have a peak brighter than pixels in this percentile.
        This helps eliminate spurious peaks. Default 64.
    margin : integer or tuple of integers, optional
        Zone of exclusion at edges of image. Default is ``separation / 2``.
    precise : boolean, optional
        Determines whether there will be an extra filtering step (``drop_close``)
        discarding features that are too close. Degrades performance.
        Because of the square kernel used, too many features are returned when
        precise=False. Default True.

    See Also
    --------
    drop_close : removes features that are too close to brighter features
    grey_dilation_legacy : local maxima finding routine used until trackpy v0.3
    """

    ndim = image.ndim
    separation = validate_tuple(separation, ndim)
    if margin is None:
        margin = tuple([int(s / 2) for s in separation])

    # Compute a threshold based on percentile.
    threshold = percentile_threshold(image, percentile)
    if np.isnan(threshold):
        #warnings.warn("Image is completely black.", UserWarning)
        logger.warning("Image is completely black.")
        return np.empty((0, ndim))

    # Find the largest box that fits inside the ellipse given by separation
    size = [int(2 * s / np.sqrt(ndim)) for s in separation]

    # The intersection of the image with its dilation gives local maxima.
    dilation = ndimage.grey_dilation(image, size, mode='constant')
    maxima = (image == dilation) & (image > threshold)
    if np.sum(maxima) == 0:
        #warnings.warn("Image contains no local maxima.", UserWarning)
        logger.warning("Image contrains no local maxima.")
        return np.empty((0, ndim))

    pos = np.vstack(np.where(maxima)).T

    # Do not accept peaks near the edges.
    shape = np.array(image.shape)
    near_edge = np.any((pos < margin) | (pos > (shape - margin - 1)), 1)
    pos = pos[~near_edge]

    if len(pos) == 0:
        #warnings.warn("All local maxima were in the margins.", UserWarning)
        logger.warning("All local maxima were in the margins")
        return np.empty((0, ndim))

    # Remove local maxima that are too close to each other
    if precise:
        pos = drop_close(pos, separation, image[maxima][~near_edge])

    return pos

def percentile_threshold(image, percentile):
    """Find grayscale threshold based on distribution in image."""

    not_black = image[np.nonzero(image)]
    if len(not_black) == 0:
        return np.nan
    return np.percentile(not_black, percentile)

def drop_close(pos, separation, intensity=None):
    """ Removes features that are closer than separation from other features.
    When intensity is given, the one with the lowest intensity is dropped:
    else the most topleft is dropped (to avoid randomness)"""
    to_drop = where_close(pos, separation, intensity)
    return np.delete(pos, to_drop, axis=0)

def where_close(pos, separation, intensity=None):
    """ Returns indices of features that are closer than separation from other
    features. When intensity is given, the one with the lowest intensity is
    returned: else the most topleft is returned (to avoid randomness)"""

    if len(pos) == 0:
        return []
    separation = validate_tuple(separation, pos.shape[1])
    if any([s == 0 for s in separation]):
        return []
    # Rescale positions, so that pairs are identified below a distance
    # of 1.
    if isinstance(pos, pd.DataFrame):
        pos_rescaled = pos.values / separation
    else:
        pos_rescaled = pos / separation
    duplicates = cKDTree(pos_rescaled, 30).query_pairs(1 - 1e-7)
    if len(duplicates) == 0:
        return []
    index_0 = np.fromiter((x[0] for x in duplicates), dtype=int)
    index_1 = np.fromiter((x[1] for x in duplicates), dtype=int)
    if intensity is None:
        to_drop = np.where(np.sum(pos_rescaled[index_0], 1) >
                           np.sum(pos_rescaled[index_1], 1),
                           index_1, index_0)
    else:
        intensity = np.asarray(intensity)
        intensity_0 = intensity[index_0]
        intensity_1 = intensity[index_1]
        to_drop = np.where(intensity_0 > intensity_1, index_1, index_0)
        edge_cases = intensity_0 == intensity_1
        if np.any(edge_cases):
            index_0 = index_0[edge_cases]
            index_1 = index_1[edge_cases]
            to_drop[edge_cases] = np.where(np.sum(pos_rescaled[index_0], 1) >
                                           np.sum(pos_rescaled[index_1], 1),
                                           index_1, index_0)
    return np.unique(to_drop)

def plot_2D_image_and_center(particle, good_fit='black'):
    """
    Plot image and centroid-found center.

    Notes:
        *
    Parameters
    ----------

    Returns
    -------

    """
    image = particle.template
    xc = particle._location_subpixel[0]
    yc = particle._location_subpixel[1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot location of center
    ax.scatter(xc, yc, s=100, marker='*', color=good_fit, alpha=0.5, label=r'$p_{xc,yc}$' +
                                '(Mass={})'.format(int(np.round(particle._fitted_centroid_on_template['mass'], -1))))

    ax.axvline(x=xc, color=good_fit, alpha=0.35, linestyle='--')
    ax.axhline(y=yc, color=good_fit, alpha=0.35, linestyle='--')

    ax.set_title(r'$p_{xc, yc}$' + '(sub-pix) = ({}, {}) \n center = {}, {}'.format(particle.location[0], particle.location[1], np.round(xc, 2), np.round(yc, 2)))

    ax.set_xlim([0, image.shape[0] - 0.5])
    ax.set_ylim([0, image.shape[1] - 0.5])
    ax.legend(fontsize=10, bbox_to_anchor=(1, 1), loc='upper left', )

    # Major ticks
    ax.set_xticks(np.arange(0, image.shape[0] + 1, 2))
    ax.set_yticks(np.arange(0, image.shape[1] + 1, 2))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(0, image.shape[0] + 1, 2))
    ax.set_yticklabels(np.arange(0, image.shape[0] + 1, 2))

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, image.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, image.shape[1], 1), minor=True)

    # invert the y-axis
    plt.gca().invert_yaxis()

    # show the image last
    ax.imshow(image, cmap='viridis', interpolation='none')

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='gray', alpha=0.125, linestyle='-', linewidth=1)
    # ax.grid(color='gray', alpha=0.125)

    return fig