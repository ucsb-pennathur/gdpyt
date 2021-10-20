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
from skimage.segmentation import clear_border, flood_fill
from skimage.measure import label, regionprops, find_contours
from skimage import (filters, io, morphology, util)
from skimage.color import label2rgb
import imutils
from skimage.filters import gaussian, median
from skimage.morphology import disk

# New imports: 9/27
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def apply_threshold(img, parameter, overlapping_particles=True, min_particle_size=5, padding=3, invert=False):
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

        """
        Options for Thresholding:
        
        #1 - Simple Threshold:
            Used in a majority of cases; always for custom synthetics and experimental datasets because no bright outer
            ring which messes up the contours. 
        #2 - Dilate, Erode, Fill Contours:
            Used for Datasets I and Datasets II when performing single particle calibration b/c it fills in the contour
            of the bright outer ring. Really only necessary for GDPT Datasets where z<-40 um.
        #3 - Erode, Dilate, Erode:
            An older method; useful when >10 pixels per particle diameter; mostly Dataset I or Dataset II.
        """

        if overlapping_particles is True:
            """
            Option #1:
            1. (9/27 - 10/9): Method used for GDPyT static and dynamic templates
            2. 10/9: Method for Single Particle Calibration on Synthetic Grid (b/c z>35 um hard to identify)
            """
            thresh_img_one = img > threshval

            thresh_img = clear_border(thresh_img_one, buffer_size=1)

        else:
            """ 
            Option #2:
            1. BEGIN NEW UPDATE: 10/9
            2. 10/10: Removing b/c working with experimental particles with high degree of overlap
            3. 10/19: Re-using to analyze calibration errors with Dataset I.
            """
            thresh_img_one = img > threshval

            # apply a large dilation to connect streaks and slightly smooth image
            dilation_size = np.min([3, min_particle_size])
            selem2 = disk(dilation_size)
            dilated = binary_dilation(thresh_img_one, selem=selem2)

            # perform an erosion to resize contour to approximate original size
            second_erosion_size = dilation_size - 1
            selem3 = disk(second_erosion_size)
            mask_closing = binary_erosion(dilated, selem=selem3)

            thresh_img = clear_border(mask_closing, buffer_size=1)

            contours = cv2.findContours(thresh_img.copy().astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contours = imutils.grab_contours(contours)

            for contour in sorted(contours, key=lambda x: x[0][0][0], reverse=False):
                M = cv2.moments(contour)
                cntr = np.max([M["m00"], 1])
                cX = int(M["m10"] / cntr)
                cY = int(M["m01"] / cntr)

                # if thresh_img[cY, cX] == 1:
                seed = (cY, cX)
                temp = flood_fill(thresh_img, seed_point=seed, new_value=1, in_place=False)
                if np.count_nonzero(temp == 1) - np.count_nonzero(thresh_img == 1) < 750:
                    thresh_img = temp

                """
                Plot to check flood filling contours
                
                fig, ax = plt.subplots()
                ax.imshow(thresh_img)
                plt.show()
                """

        """
        Option #3:
        1. Up to 9/27: Old threshold method - Removed to focus on scikit-image segmentation methods.
        2. 11/9: Reusing for GDPT Dataset II
        3. 11/10: Stopped using for Synthetic Grid Overlap because reduces the number of particles identified

        noisy_thresh_image = img > threshval

        # apply a small erosion to remove bright spots smaller than particle diameter
        first_erosion_size = np.min([3, int(np.ceil(min_particle_size/3))])
        selem1 = disk(first_erosion_size)
        eroded = binary_erosion(noisy_thresh_image, selem=selem1)

        # apply a large dilation to connect streaks and slightly smooth image
        dilation_size = np.min([5, min_particle_size])
        selem2 = disk(dilation_size)
        dilated = binary_dilation(eroded, selem=selem2)

        # perform an erosion to resize contour to approximate original size
        second_erosion_size = dilation_size - first_erosion_size
        selem3 = disk(second_erosion_size)
        mask_closing = binary_erosion(dilated, selem=selem3)

        # clear the pixel values on border
        thresh_img = clear_border(mask_closing, buffer_size=1)
        """

        """fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(noisy_thresh_image)
        ax[1].imshow(thresh_img)
        plt.show()"""


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

def identify_contours_sk(particle_mask, intensity_image, same_id_threshold, overlapping_particles, filename=None):

    if overlapping_particles is True:
        # separate connected contours and label image
        distance = ndi.distance_transform_edt(particle_mask)
        coords = peak_local_max(image=distance, min_distance=5, footprint=np.ones((3, 3)), labels=particle_mask)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=particle_mask)
        label_image = label(labels)
    else:
        # label the particle mask without segmentation
        label_image = label(particle_mask)

    # Plot to check the effectiveness of image segmentation
    """
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(particle_mask, cmap=plt.cm.gray)
    ax[0].set_title('Overlapping objects')

    if overlapping_particles is True:
        ax[1].imshow(labels, cmap=plt.cm.nipy_spectral)
    else:
        image_label_overlay = label2rgb(label_image, image=intensity_image, bg_label=0)
        ax[1].imshow(image_label_overlay)
        ax[1].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()
    """

    # region properties
    regions = regionprops(label_image=label_image, intensity_image=intensity_image)

    # check if any regions are within the same_id_threshold and remove regions if max/mean intensity is much lower
    labels = []
    weighted_centroids = []
    max_intensities = []
    mean_intensities = []
    for region in regions:
        labels.append(region.label)
        weighted_centroids.append(region.weighted_centroid)
        max_intensities.append(region.max_intensity)
        mean_intensities.append(region.mean_intensity)

    labels_to_remove = []
    for lbl, wc, maxi, meani in zip(labels, weighted_centroids, max_intensities, mean_intensities):
        for lbl_i, wc_i, maxi_i, meani_i in zip(labels, weighted_centroids, max_intensities, mean_intensities):
            if lbl_i != lbl:
                if np.all([np.abs(wc[0] - wc_i[0]) < same_id_threshold, np.abs(wc[1] - wc_i[1]) < same_id_threshold]):
                    if maxi_i < maxi * 0.85:
                        labels_to_remove.append(lbl_i)

    """logger.debug("{} contours in thresholded image".format(len(contours)))
    contours, bboxes = self.merge_overlapping_particles(contours, bboxes, overlap_thresh=overlap_threshold)
    logger.debug("{} contours in thresholded image after merging of overlapping".format(len(contours)))"""

    # get the contours for each region
    contour_coords = []
    new_regions = []

    for region in regions:
        if region.label not in labels_to_remove:
            new_regions.append(region)

            zero_array = np.zeros_like(intensity_image.T, dtype=int)
            points = region.coords
            zero_array[points[:, 1], points[:, 0]] = 1

            cont = find_contours(zero_array)

            if len(cont) == 0:
                continue
            else:
                contour = cont[0].astype(int)
                contour_coords.append(contour)

            """if filename in ['calib_-14.0.tif', 'calib_-13.0.tif']:
                fig, ax = plt.subplots()
                ax.imshow(zero_array.T)
                ax.plot(contour[:, 0], contour[:, 1], color='red', linewidth=1)
                plt.show()
                j=1
                plt.close()"""


    return label_image, new_regions, contour_coords

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