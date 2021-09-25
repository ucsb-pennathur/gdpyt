import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.interpolate import splprep, splev
from skimage.filters import threshold_mean, threshold_minimum, threshold_triangle
from skimage.filters import threshold_otsu, threshold_multiotsu, threshold_local
from skimage.filters import threshold_niblack, threshold_li, threshold_sauvola
from skimage.morphology import disk, white_tophat, closing, square, binary_closing, binary_dilation, binary_erosion
from skimage.filters.rank import mean_bilateral
from skimage.exposure import equalize_adapthist
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage import (
    color, feature, filters, io, measure, morphology, segmentation, util
)
import imutils
from skimage.filters import gaussian, median
from skimage.morphology import disk

def apply_threshold(img, parameter, min_particle_size=5, padding=3, invert=False):
    if not len(parameter) == 1:
        raise ValueError("Thresholding parameter must be specified as a dictionary with one key and a list containing"
                         "supplementary arguments for that function as a value")

    method = list(parameter.keys())[0]
    if method not in ['none', 'otsu', 'multiotsu', 'local', 'min', 'mean', 'mean_percent', 'median', 'median_percent',
                      'manual', 'manual_percent', 'triangle', 'manual_smoothing', 'li', 'niblack', 'sauvola']:
        raise ValueError("method must be one of ['none', otsu', 'multiotsu', 'local', 'min',"
                         " 'manual', 'triangle', 'manual_smoothing', 'li', 'niblack', 'sauvola']")

    if method in ['manual', 'manual_percent', 'manual_smoothing']:
        manual_initial_guess = np.round(img.mean() + np.std(img) * parameter[method][0], 0)

    if method == 'none':
        thresh_val = 0
        thresh_img = img
    elif method == 'otsu':
        thresh_val = threshold_otsu(img)
        bw = closing(img > thresh_val, square(3))
        thresh_img = clear_border(bw)
        # thresh_img = img > thresh_val -- old: updated 8/14/21
    elif method == 'multiotsu':
        kwargs = parameter[method]
        thresh_val = threshold_multiotsu(img, **kwargs)
        thresh_img = np.digitize(img, bins=thresh_val) # edit (10/11/20)
        #thresh_img = img > thresh_val[-1]               # original
    elif method == 'mean':
        thresh_val = threshold_mean(img)
        thresh_img = img > thresh_val
    elif method == 'mean_percent':
        thresh_val = threshold_mean(img) + threshold_mean(img) * parameter[method][0]
        thresh_img = img > thresh_val
    elif method == 'median':
        thresh_val = np.median(img)
        bw = closing(img > thresh_val, square(3))
        thresh_img = clear_border(bw)
        # thresh_img = img > thresh_val -- old: updated 8/14/21
    elif method == 'median_percent':
        thresh_val = np.median(img) + np.median(img) * parameter[method][0]
        bw = closing(img > thresh_val, square(3)) # TODO: this closing probably removes small particles that i would rather keep
        thresh_img = clear_border(bw)
        # thresh_img = img > thresh_val -- old: updated 8/14/21
    elif method == 'min':
        thresh_val = threshold_minimum(img)
        thresh_img = img > thresh_val
    elif method == 'local':
        kwargs = parameter[method]
        assert isinstance(kwargs, dict)
        thresh_val = threshold_local(img, **kwargs)
        thresh_img = img > thresh_val
    elif method == 'triangle':
        thresh_val = threshold_triangle(img)
        thresh_img = img > thresh_val
    elif method == 'manual_smoothing':
        threshval = manual_initial_guess
        dividing = img > threshval
        smoother_dividing = filters.rank.mean(util.img_as_ubyte(dividing), morphology.disk(parameter[method][0]))
        thresh_img = smoother_dividing > parameter[method][1]
    elif method == 'li':
        threshval = threshold_li(img, initial_guess=threshold_otsu(img))
        thresh_img = img > threshval
    elif method == 'niblack':
        kwargs = parameter[method]
        assert isinstance(kwargs, dict)
        threshval = threshold_niblack(img, **kwargs)
        thresh_img = img > threshval
    elif method == 'sauvola':
        kwargs = parameter[method]
        assert isinstance(kwargs, dict)
        R = img.std()
        threshval = threshold_sauvola(img, r=R, **kwargs)
        thresh_img = img > threshval
    elif method == 'manual_percent':
        thresh_val = parameter[method][0] + parameter[method][0] * parameter[method][1]
        bw = closing(img > thresh_val, square(3))
        thresh_img = clear_border(bw)
    elif method == 'manual':
        args = parameter[method]
        if not isinstance(parameter[method], list):
            threshval = parameter[method]
        elif len(parameter[method]) == 1:
            threshval = parameter[method]
        else:
            if not len(parameter[method]) == 1:
                raise ValueError("For manual thresholding only one parameter (the manual threshold) must be specified")
            threshval = manual_initial_guess

        # threshold the image
        noisy_thresh_image = img > threshval

        # apply a small erosion to remove bright spots smaller than particle diameter
        first_erosion_size = np.min([3, int(np.ceil(min_particle_size/3))])
        selem1 = disk(first_erosion_size)
        eroded = binary_erosion(noisy_thresh_image, selem=selem1)

        # apply a large dilation to connect streaks and slightly smooth image
        dilation_size = np.min([7, min_particle_size])
        selem2 = disk(dilation_size)
        dilated = binary_dilation(eroded, selem=selem2)

        # perform an erosion to resize contour to approximate original size
        second_erosion_size = dilation_size - first_erosion_size
        selem3 = disk(second_erosion_size)
        mask_closing = binary_erosion(dilated, selem=selem3)

        # clear the pixel values on border
        thresh_img = clear_border(mask_closing, buffer_size=1)

    if invert:
        thresh_img = ~thresh_img

    return thresh_img

def identify_contours(particle_mask):
    """
    Notes
    * OpenCV is used here because it provides the integer-valued coordinates of the contours. Skimage provides the
    floating point value between YES/NO contour pixels and, for this reason, it is not suitable here.
    * OpenCV uses a (y, x) convention in arrays (I'm pretty sure).
    """
    # this is equivalent to a binary closing which can connect small bright cracks
    """
    selem1 = disk(2)
    selem2 = disk(6)
    selem3 = disk(3)
    eroded = binary_erosion(particle_mask, selem=selem1)
    dilated = binary_dilation(eroded, selem=selem2)
    mask_closing = binary_erosion(dilated, selem=selem3)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(11,3))
    ax1.imshow(particle_mask)
    ax1.set_title('particle mask')
    ax2.imshow(eroded)
    ax2.set_yticks([])
    ax2.set_title('erosion')
    ax3.imshow(mask_closing)
    ax3.set_yticks([])
    ax3.set_title('eroded, dilate, erode')
    ax4.imshow(selem2)
    ax4.set_yticks([])
    ax4.set_title('filter')
    plt.show()
    """

    #contours = cv2.findContours(particle_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = cv2.findContours(particle_mask.copy().astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    bboxes = [cv2.boundingRect(contour) for contour in contours]

    return contours, bboxes

def identify_circles(image):
    h, w = image.shape
    min_radius = 3
    circles = cv2.HoughCircles(image.copy(), cv2.HOUGH_GRADIENT, dp=1.1, minDist=min_radius * 2,
                               param1=20, param2=5, minRadius=min_radius, maxRadius=int(np.minimum(h, w) / 2))
    # Create contours
    n_pts = 360
    contours = []
    d_angle = 2 * np.pi / n_pts

    if circles is not None:
        for circle in circles[0]:
            center = np.array([[circle[0], circle[1]]])
            contour = [center + circle[2]*np.array([[np.cos(i * -d_angle), np.sin(i * -d_angle)]])
                       for i in range(n_pts)]
            contour = np.unique(np.array(contour).astype(np.int), axis=0)
            contours.append(contour)

        bboxes = [cv2.boundingRect(contour) for contour in contours]
    else:
        bboxes = []

    return contours, bboxes

def merge_particles(particles):
    id_ = None
    merged_contour = None
    for particle in particles:
        if id_ is None:
            id_ = particle.id
        else:
            if not particle.id == id_:
                raise ValueError("Only particles with duplicate IDs can be merged")

        if merged_contour is None:
            merged_contour = particle.contour.copy()
        else:
            merged_contour = np.vstack([merged_contour, particle.contour.copy()])

    new_contour = cv2.convexHull(merged_contour)
    new_bbox = cv2.boundingRect(new_contour)

    return new_contour, new_bbox

def binary_mask(radius, ndim):
    "Elliptical mask in a rectangular array from TrackPy's masks.py"
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