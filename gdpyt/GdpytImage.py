import cv2
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from skimage import io
from skimage.filters import median, gaussian
from skimage.morphology import disk, white_tophat, closing, square
from skimage.filters.rank import mean_bilateral
# TODO: investigate the below functions
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio

from skimage.exposure import equalize_adapthist
from skimage.segmentation import clear_border as sk_clear_borders
from skimage.measure import label, regionprops, regionprops_table, find_contours
from skimage.draw import rectangle_perimeter, polygon

import numpy as np
import numpy.ma as ma
import pandas as pd
from .particle_identification import apply_threshold, identify_contours, identify_contours_sk, merge_particles
from .GdpytParticle import GdpytParticle
from gdpyt.similarity.correlation import *
from os.path import isfile, basename
import time
import logging

logger = logging.getLogger()

class GdpytImage(object):
    """
    This class holds an image along with it's properties such as the
    raw image, filtered image, path, filename, particles present in the image. If the image is part of a calibration,
    the z coordinate is passed when creating an instance
    """

    def __init__(self, path, frame=None, if_img_stack_take='all', take_subset_mean=None, true_num_particles=None):
        super(GdpytImage, self).__init__()
        # Attributes with an underscore as the first character are "internal use". That means that they are only meant
        # to be modified by methods of this class.
        # The reasoning behind this is the following: Imagine an image with a number of particles. The particles are saved
        # in the self._particles attribute. We only want the identify_particles method in this class to modify this attribute.
        # We don't really want any other part of the program to change this attribute, since the particles in an image
        # are always going to be those that were identified with identify_particles. If this attribute isn't marked as
        # "internal use", other parts of the program could by accidentally add or delete particles.

        if not isfile(path):
            raise ValueError("{} is not a valid file".format(path))

        self._filepath = path
        self._filename = basename(path)
        self.frame = frame
        if true_num_particles is None:
            self._true_num_particles = 1
        else:
            self._true_num_particles = true_num_particles

        # Load the image. This sets the ._raw attribute
        self.load(path, if_img_stack_take=if_img_stack_take, take_subset_mean=take_subset_mean)

        # Crop the image. This sets the ._original attribute
        self._original = None

        # Background image
        self._subbg = None

        # Filtered image. This attribute is assigned by using the filter_image method
        self._filtered = None
        self._processing_stats = None
        self._masked = None

        # Particles: dictionary {particle_id: Particle object}
        # This dictionary is filled up with the identify_particles method
        self._particles = []
        self._particle_similarity = None
        self._z = None

        # contour stats
        self.regionprops_table = None
        self._mean_contour_area = None
        self._std_contour_area = None

    def __repr__(self):
        class_ = 'GdpytImage'
        repr_dict = {'Dimensions': self.shape,
                     'Particles in this image': [particle.id for particle in self.particles],
                     'Z coordinate': self._z}
        out_str = "{}: {} \n".format(class_, self.filename)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def _add_particle(self, id_, contour, bbox, particle_mask, particle_collection_type=None, location=None):
        self._particles.append(GdpytParticle(self.raw, self.filtered, id_, contour, bbox,
                                             particle_mask_on_image=particle_mask,
                                             particle_collection_type=particle_collection_type,
                                             location=location, frame=self.frame))

    def _update_processing_stats(self, names, values):
        if not isinstance(names, list):
            names = [names]
        new_stats = {}
        for name, value in zip(names, values):
            new_stats.update({name: [value]})
        new_stats = pd.DataFrame(new_stats)

        if self._processing_stats is None:
            self._processing_stats = new_stats
        else:
            self._processing_stats = new_stats.combine_first(self._processing_stats)


    def draw_particles(self, raw=True, thickness=2, draw_id=True, draw_bbox=True):
        if raw:
            canvas = self.raw.copy()
        else:
            canvas = self.filtered.copy()

        max_val = int(canvas.max())
        color = (max_val, max_val, max_val)
        for particle in self.particles:
            cv2.drawContours(canvas, [particle.contour], -1, color, thickness)
            if draw_id:
                bbox = particle.bbox
                coords = (int(bbox[0] - 0.2 * bbox[2]), int(bbox[1] - 0.2 * bbox[3]))
                cv2.putText(canvas, "ID: {}".format(particle.id), coords, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            if draw_bbox:
                x, y, w, h = particle.bbox
                cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

        return canvas

    def crop_image(self, cropspecs):
        """
        This crops the image. The argument cropspecs is a dictionary of xmin, xmax, ymin, ymax and values.
        :param cropspecs:
        """
        valid_crops = ['xmin', 'xmax', 'ymin', 'ymax', 'pad']

        for crop_func in cropspecs.keys():
            if crop_func not in valid_crops:
                raise ValueError("{} is not a valid crop dimension. Use: {}".format(crop_func, valid_crops))

        if self._original is None:
            self._original = self._raw.copy()
        self._raw = self._original[cropspecs['ymin']:cropspecs['ymax'], cropspecs['xmin']:cropspecs['xmax']]

        if 'pad' in cropspecs.keys():
            self._raw = np.pad(self._raw, pad_width=cropspecs['pad'], mode='constant', constant_values=np.min(self._raw))

    def subtract_background(self, background_subtraction, background_img):
        """
        This subtracts the background image from each image in collection.
        :param background_subtraction:
        :return:

        This method should assign self._subbg
        """
        valid_bs_methods = ['min']

        if background_subtraction not in valid_bs_methods:
            raise ValueError("{} is not a valid method. Implemented so far are {}".format(background_subtraction,
                                                                                          valid_bs_methods))
            self._subbg = None
        else:
            img = self._raw.copy()
            self._subbg = img - background_img



    def filter_image(self, filterspecs, force_rawdtype=True):
        """
        Steps:
            1. clear borders

        This is an image filtering function. The argument filterdict are similar to the arguments of the
        image_smoothing function.
        e.g. filterdict: {'median': 5, 'bilateral': 4, 'gaussian': 5}
        :param filterdict:
        :return:

        This method should assign self._processed and self._processing_stats
        """
        img_copy = self._raw.copy()
        raw_dtype = img_copy.dtype

        valid_filters = ['none', 'median', 'mean_bilateral', 'gaussian', 'white_tophat', 'equalize_adapthist']
        #TODO: - there are several 'denoising' filters that would be useful to investigate.
        #   > particularly the skimage.restoration.denoise_bilateral function which perserves edges.
        #   > see more here: https://scikit-image.org/docs/dev/auto_examples/filters/plot_denoise.html#sphx-glr-auto-examples-filters-plot-denoise-py
        #   > there are very exciting results that perserve texture shown in the example below:
        #   > see here: https://scikit-image.org/docs/dev/auto_examples/filters/plot_nonlocal_means.html#sphx-glr-auto-examples-filters-plot-nonlocal-means-py

        if len(filterspecs) == 0:
            img = img_copy
        for process_func in filterspecs.keys():
            if process_func not in valid_filters:
                raise ValueError("{} is not a valid filter. Implemented so far are {}".format(process_func, valid_filters))
            if process_func == "none":
                img = img_copy
            else:
                func = eval(process_func)
                args = filterspecs[process_func]['args']
                if 'kwargs' in filterspecs[process_func].keys():
                    kwargs = filterspecs[process_func]['kwargs']
                else:
                    kwargs = {}

                img = apply_filter(img_copy, func, *args, **kwargs)
                if process_func == "equalize_adapthist":
                    img = img*img_copy.max()
                if force_rawdtype and img.dtype != raw_dtype:
                    img = img.astype(raw_dtype)

        self._filtered = img

    def get_particle(self, id_):
        ret_particle = []
        for particle in self.particles:
            if particle.id == id_:
                ret_particle.append(particle)
        if len(ret_particle) == 0:
            logger.error("No particle with ID {} found in this image".format(id_))
        elif len(ret_particle) > 1:
            logger.warning("In image {}, {} particles with ID {} were found".format(self.filename, len(ret_particle), id_))

        return ret_particle

    def identify_particles_sk(self, thresh_specs, min_size=None, max_size=None, shape_tol=0.1,
                              overlap_threshold=0.3, same_id_threshold=10,
                              padding=2, inspect_contours_for_every_image=False, image_collection_type=None,
                              particle_id_image=None):
        """
        Method:
            1. apply threshold
            2. Identify contours
            3. Sort contours:
                3.1. if min_siz < area < max_size
                3.2. if aspect ratio is not square
                3.3. if thinness ratio is not a circle
                3.4. if too close to the image borders

        """

        if shape_tol is not None:
            assert 0 < shape_tol < 1

        if particle_id_image is not None:
            """ Using static templates in this case """
            particle_mask = apply_threshold(particle_id_image, parameter=thresh_specs, min_particle_size=min_size,
                                            padding=padding).astype(np.uint16)
        else:
            """ Using dynamic templates in this case """
            particle_mask = apply_threshold(self.filtered, parameter=thresh_specs, min_particle_size=min_size,
                                            padding=padding).astype(np.uint16)

        """if self.filename in ['B00051.tif', 'B00052.tif', 'B00053.tif', 'B00054.tif']:
            fig, ax = plt.subplots(ncols=2)
            ax[0].imshow(self.filtered)
            ax[1].imshow(particle_mask)
            plt.show()"""

        # identify particles
        label_image, regions, all_contour_coords = identify_contours_sk(particle_mask, self.filtered, same_id_threshold,
                                                                        self.filename)

        """logger.debug("{} contours in thresholded image".format(len(all_contour_coords)))
        contours, bboxes = self.merge_overlapping_particles(all_contour_coords, bboxes, overlap_thresh=overlap_threshold)
        logger.debug("{} contours in thresholded image after merging of overlapping".format(len(contours)))"""

        # store the regionprops table
        regionprops_data = regionprops_table(label_image, self.filtered,
                                properties=['label', 'area', 'bbox', 'centroid', 'weighted_centroid', 'local_centroid',
                                            'weighted_local_centroid', 'max_intensity', 'mean_intensity',
                                            'minor_axis_length', 'major_axis_length'])
        regionprops_data = pd.DataFrame(regionprops_data)
        self.regionprops_table = regionprops_data

        # filters regions (contours)
        skipped_contours = []
        passing_ids = []
        contour_areas = []
        id_ = 0

        # Sort contours and bboxes by x-coordinate:
        for region, contour_coords in sorted(zip(regions, all_contour_coords), key=lambda x: x[0].centroid[1]):

            # filter on area
            area = region.area
            if area < min_size or area > max_size:
                skipped_contours.append(region.label)
                continue

            # filter on aspect ratio
            if region.minor_axis_length < 2:
                logger.warning("Region skipped b/c minor axis length {} < 2".format(aspect_ratio))
                skipped_contours.append(region.label)
                continue
            aspect_ratio = region.major_axis_length / region.minor_axis_length
            if aspect_ratio > 6:
                logger.warning("Region skipped b/c aspect ratio = {} > 3.".format(aspect_ratio))
                skipped_contours.append(region.label)
                continue

            # filter on circularity
            solidity = region.solidity
            """if solidity < 0.5:
                skipped_contours.append(region.label)
                continue"""

            # filter on eccentricity
            eccentricity = region.eccentricity
            """if eccentricity > 0.5:
                skipped_contours.append(region.label)
                continue"""

            # adjust the bounding box (bbox) to work with GdpytParticle (note: x0, y0, w0, h0 = self.bbox)
            min_row, min_col, max_row, max_col = region.bbox
            bbox = (min_col, min_row, max_col - min_col, max_row - min_row)

            # find the contour of the region slice
            # contour_coords = find_contours(region.filled_image)
            """h = region.coords
            h1, h2 = h[:, 0], h[:, 1]
            contour_coords = np.vstack((h2, h1)).T"""

            cX = int(np.round(regionprops_data[regionprops_data['label'] == region.label]['weighted_centroid-1'].item(), 0))
            cY = int(np.round(regionprops_data[regionprops_data['label'] == region.label]['weighted_centroid-0'].item(), 0))

            bbox, bbox_center = self.pad_and_center_region(cX, cY, bbox, padding=padding)

            # discard contours that are too close to the image borders to include the desired padding
            if bbox[0] - padding < 1 or bbox[1] - padding < 1:
                skipped_contours.append(region.label)
                print("Skipped because template + padding near the image borders")
                continue
            elif bbox[0] + bbox[2] + padding >= self.shape[1] or bbox[1] + bbox[3] + padding >= self.shape[0]:
                skipped_contours.append(region.label)
                print("Skipped because template + padding near the image borders")
                continue

            # Add data for contour inspection
            contour_areas.append(area)
            passing_ids.append(id_)

            # option to plot identified contours and bounding boxes
            """show_plots = False
            if show_plots is True:
                fig, ax = plt.subplots(figsize=(6,2))
                ax.imshow(self._filtered)
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                ax.set_title("Contour ID {}, bbox (x, y, w, h) = ({})".format(id_, bbox))
                plt.suptitle(self.filename)
                savedir = '/Users/mackenzie/Desktop/dumpfigures/particle_segmentation'
                plt.savefig(join(savedir, '{}_contourID{}.png'.format(self.filename[:-4], id_)))
                plt.show()
                plt.close()

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(region.intensity_image)
                ax.set_title("{}, contour ID {}".format(self.filename, id_))
                savedir = '/Users/mackenzie/Desktop/dumpfigures/particle_segmentation'
                plt.savefig(join(savedir, '{}_template_contourID{}.png'.format(self.filename[:-4], id_)))
                plt.show()
                plt.close()"""

            # Add particle
            self._add_particle(id_, contour_coords, bbox, particle_mask, particle_collection_type=image_collection_type,
                               location=(cX, cY))
            id_ = id_ + 1


        # Calculate contour statistics
        if len(contour_areas) > 0:
            min_contour_area = np.min(contour_areas)
            max_contour_area = np.max(contour_areas)
            mean_contour_area = np.mean(contour_areas)
            std_contour_area = np.round(np.std(contour_areas), 1)
            if min_contour_area < min_size * 1.5:
                print("Min. contour area [input, measured] = [{}, {}]".format(min_size, min_contour_area))
                print("Mean contour area: {} +/- {}".format(mean_contour_area, std_contour_area * 2))
            if max_contour_area > max_size * 0.75:
                print("Max. contour area [input, measured] = [{}, {}]".format(max_size, max_contour_area))
                print("Mean contour area: {} +/- {}".format(mean_contour_area, std_contour_area * 2))

            # set image attributes
            self._masked = particle_mask
            self._mean_contour_area = mean_contour_area
            self._std_contour_area = std_contour_area

            # calculate image statistics
            self.calculate_image_statistics()

    def identify_particles(self, thresh_specs, min_size=None, max_size=None, shape_tol=0.1, overlap_threshold=0.3,
                           padding=2, inspect_contours_for_every_image=False, image_collection_type=None):
        if shape_tol is not None:
            assert 0 < shape_tol < 1
        particle_mask = apply_threshold(self.filtered, parameter=thresh_specs, min_particle_size=min_size, padding=padding).astype(np.uint16)

        # Identify particles
        contours, bboxes = identify_contours(particle_mask.astype(np.uint8))
        logger.debug("{} contours in thresholded image".format(len(contours)))
        contours, bboxes = self.merge_overlapping_particles(contours, bboxes, overlap_thresh=overlap_threshold)
        logger.debug("{} contours in thresholded image after merging of overlapping".format(len(contours)))

        # remove particles that are too close together
        # contours, bboxes = self.remove_nearby_particles(contours, bboxes, threshold_distance=10)

        # storage lists
        skipped_cnts = []
        passing_cnts = []
        passing_bboxes = []
        passing_ids = []
        contour_areas = []

        id_ = 0
        # Sort contours and bboxes by x-coordinate:
        for cont_bbox in sorted(zip(contours, bboxes), key=lambda b: b[1][0], reverse=True):

            contour = cont_bbox[0]

            # get contour area and perimeter
            contour_area = abs(cv2.contourArea(contour))
            contour_perim = cv2.arcLength(contour, True)

            # If specified, check if contour is too small or too large. If true, skip the creation of the particle
            if min_size is not None:
                if contour_area < min_size:
                    skipped_cnts.append(contour)
                    print("Skipped because contour area {} < {} threshold".format(contour_area, min_size))
                    continue
            if max_size is not None:
                if contour_area > max_size:
                    skipped_cnts.append(contour)
                    print("Skipped because contour area {} > {} threshold".format(contour_area, max_size))
                    continue

            bbox = cont_bbox[1]

            if shape_tol is not None:
                # Discard contours that are clearly not a circle just by looking at the aspect ratio of the bounding box
                bbox_ar = bbox[2] / bbox[3]
                bbox_aspect_ratio = abs(np.maximum(bbox_ar, 1 / bbox_ar) - 1)
                if bbox_ar < shape_tol:
                    skipped_cnts.append(contour)
                    print("Skipped because bbox aspect ratio {} < {} shape tolerance".format(bbox_ar, shape_tol))
                    continue
                # Check if circle by calculating thinness ratio
                bbox_tr = 4 * np.pi * contour_area / contour_perim**2
                bbox_thinness = abs(np.maximum(bbox_tr, 1 / bbox_tr) - 1)
                if bbox_tr < shape_tol:
                    #skipped_cnts.append(contour)
                    #print("Skipped because bbox thinness {} < {} shape tolerance".format(bbox_tr, shape_tol))
                    #continue
                    print("Would've skipped because bbox thinness {} < {} shape tolerance but thinness filter is off".format(bbox_tr, shape_tol))

            bbox, bbox_center = self.pad_and_center_contour(contour, bbox, padding=padding)

            # discard contours that are too close to the image borders to include the desired padding
            if bbox[0] - padding < 1 or bbox[1] - padding < 1:
                skipped_cnts.append(contour)
                print("Skipped because template + padding near the image borders")
                continue
            elif bbox[0] + bbox[2] + padding >= self.shape[1] or bbox[1] + bbox[3] + padding >= self.shape[0]:
                h = bbox[0] + bbox[2] + padding
                j = bbox[1] + bbox[3] + padding
                skipped_cnts.append(contour)
                print("Skipped because template + padding near the image borders")
                continue

            # Add data for contour inspection
            contour_areas.append(contour_area)
            passing_cnts.append(contour)
            passing_bboxes.append(bbox)
            passing_ids.append(id_)

            # Add particle
            self._add_particle(id_, contour, bbox, particle_mask, particle_collection_type=image_collection_type,
                               location=bbox_center)
            id_ = id_ + 1

        # Code to plot all the contours and bounding boxes for every image
        """
        if inspect_contours_for_every_image:
            img = np.zeros_like(self.raw, dtype=np.uint16)
            for con_box in sorted(zip(contours, bboxes), key=lambda b: b[1][0], reverse=True):
                cntr = con_box[0]
                cntr = np.squeeze(cntr)
                if np.size(cntr) > 3:
                    bbx = con_box[1]
                    x, y, w, h = bbx
                    rr, cc = rectangle_perimeter(start=(x, y), end=(x + w, y + h), shape=self.raw.shape)
                    img[cc, rr] = 2 ** 11
                    rr, cc = polygon(cntr[:,0], cntr[:,1], img.shape)
                    img[cc, rr] = 2**10
            for con_box in sorted(zip(passing_cnts, passing_bboxes, passing_ids), key=lambda b: b[1][0], reverse=True):
                cntr = con_box[0]
                cntr = np.squeeze(cntr)
                if np.size(cntr) > 3:
                    bbx = con_box[1]
                    x, y, w, h = bbx
                    rr, cc = rectangle_perimeter(start=(x, y), end=(x + w, y + h), shape=self.raw.shape)
                    img[cc, rr] = 2 ** 13
                    rr, cc = polygon(cntr[:,0], cntr[:,1], img.shape)
                    img[cc, rr] = 2**11
                    cv2.putText(img, "ID: {}".format(con_box[2]), [x, y], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (2 ** 13, 2 ** 13, 2 ** 13), 2)
            fig, ax = plt.subplots(ncols=3, figsize=(12,4))
            ax[0].imshow(self.filtered, cmap='viridis')
            ax[0].set_title('filtered')
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('y')
            ax[1].imshow(self.masked, cmap='viridis')
            ax[1].set_title('masked')
            ax[1].axis('off')
            ax[2].imshow(img, cmap='viridis')
            ax[2].set_title('{} - mode=wrap'.format(self.filename))
            ax[2].axis('off')
            plt.show()
            """

        # Calculate contour statistics
        if len(contour_areas) > 0:
            min_contour_area = np.min(contour_areas)
            max_contour_area = np.max(contour_areas)
            mean_contour_area = np.mean(contour_areas)
            std_contour_area = np.round(np.std(contour_areas), 1)
            if min_contour_area < min_size * 1.5:
                print("Min. contour area [input, measured] = [{}, {}]".format(min_size, min_contour_area))
                print("Mean contour area: {} +/- {}".format(mean_contour_area, std_contour_area * 2))
            if max_contour_area > max_size * 0.75:
                print("Max. contour area [input, measured] = [{}, {}]".format(max_size, max_contour_area))
                print("Mean contour area: {} +/- {}".format(mean_contour_area, std_contour_area * 2))

        # set image attributes
        self._masked = particle_mask
        self._mean_contour_area = mean_contour_area
        self._std_contour_area = std_contour_area

        # calculate image statistics
        self.calculate_image_statistics()

    def calculate_image_statistics(self):

        # initialize variables
        particle_mask = self._masked
        mean_contour_area = self._mean_contour_area
        std_contour_area = self._std_contour_area

        # Calculate image statistics
        background_mask = particle_mask.astype(bool)

        # copy image numpy arrays for manipulation while perserving the original data
        img_f = self.filtered.copy()
        img_r = self.raw.copy()
        img_f_bkg = self.filtered.copy()
        img_r_bkg = self.raw.copy()

        # apply background mask to get background
        img_f_mask_inv = ma.masked_array(img_f, mask=particle_mask)
        img_r_mask_inv = ma.masked_array(img_r, mask=particle_mask)

        # apply particle mask to get signal
        particle_mask = np.logical_not(background_mask).astype(bool)
        img_f_mask = ma.masked_array(img_f_bkg, mask=particle_mask)
        img_r_mask = ma.masked_array(img_r_bkg, mask=particle_mask)

        # calculate SNR for raw image
        mean_signal_r = img_r_mask.mean()
        mean_background_r = img_r_mask_inv.mean()
        std_background_r = img_r_mask_inv.std()
        snr_raw = (mean_signal_r - mean_background_r) / std_background_r

        # calculate SNR for filtered image
        mean_signal_f = img_f_mask.mean()
        mean_background_f = img_f_mask_inv.mean()
        std_background_f = img_f_mask_inv.std()

        # background can be zero for noise=0 calibration datasets so make == 1 for simplicity
        if std_background_f < 1:
            std_background_f = 1

        snr_filtered = (mean_signal_f - mean_background_f) / std_background_f
        """
        Signal-to-noise ratio (Barnkob & Rossi 2020): SNR = mean particle image signal / standard deviation of noise
        """

        # the snr is capped at 250 b/c greater than this is meaningless
        if snr_filtered > 250:
            snr_filtered = 250

        # calculate pixel density (# of pixels in particles / # of pixels in image) for filtered image
        pixel_density = background_mask.sum() / background_mask.size
        """
        Particle image density (Barnkob & Rossi 2020): Density = sum of particle image areas / full image area
        """

        # calculate particle density (# of particles / # of pixels in image) for filtered image
        if self.particles:
            num_particles = len(self.particles)
            particle_density = num_particles / background_mask.size
            percent_particles_idd = num_particles / self.true_num_particles
            self._update_processing_stats(
                ['mean_signal', 'mean_background', 'std_background', 'snr_filtered', 'pixel_density',
                 'particle_density', 'percent_particles_idd', 'true_num_particles', 'contour_area_mean', 'contour_area_std'],
                [mean_signal_f, mean_background_f, std_background_f, snr_filtered, pixel_density, particle_density,
                 percent_particles_idd, self.true_num_particles, mean_contour_area, std_contour_area])
        else:
            raise ValueError("why are there zero particles for this image?")
            #TODO: add statistics for contours that did not pass--so I can understand why they didn't pass.
            self._update_processing_stats(
                ['mean_signal', 'mean_background', 'std_background', 'snr_filtered', 'pixel_density',
                 'particle_density', 'percent_particles_idd', 'true_num_particles'],
                [mean_signal_f, mean_background_f, std_background_f, snr_filtered, pixel_density, 0.0, 0.0,
                 self.true_num_particles])

        """
        If you want to plot the particle/background images
            NOTE - you must use interpolation='none' in order to properly see the image.
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(img_f_mask, cmap='gray', interpolation='none')
        ax[1].imshow(img_f_mask_inv, cmap='gray', interpolation='none')
        plt.show()
        """


    def _update_particle_density_stats(self):
        num_particles = len(self.particles)
        particle_density = num_particles / self.raw.size
        percent_particles_idd = num_particles / self.true_num_particles

        self._update_processing_stats(['particle_density', 'percent_particles_idd', 'true_num_particles'],
            [particle_density, percent_particles_idd, self.true_num_particles])

    def is_infered(self):
        return all([particle.z is not None for particle in self.particles])

    def load(self, path=None, if_img_stack_take='mean', take_subset_mean=None):
        # load using skimage
        img = io.imread(self._filepath, plugin='tifffile')

        # check if image is a stack
        if len(np.shape(img)) > 2:   # image is a stack
            if if_img_stack_take == 'mean':
                img = np.rint(np.mean(img, axis=0, dtype=float)).astype(np.int16)
            elif if_img_stack_take == 'first':
                img = img[0, :, :]
            elif if_img_stack_take == 'subset':
                img = np.rint(np.mean(img[take_subset_mean[0]:take_subset_mean[1], :, :], axis=0, dtype=float)).astype(np.int16)
            else:
                raise ValueError("if_img_stack_take must equal 'mean', 'first', or 'subset'. If 'subset', take_subset_mean must be a 2-item tuple or list indicating a START and STOP index.")

        self._original = img
        self._raw = self._original


    def read_tiff_tag(self):
        """
        Link: https://github.com/blink1073/tifffile/blob/master/tifffile/tifffile.py

        Returns
        -------

        """
        from tifffile import TiffFile
        # iterate over pages and tags in TIFF file
        with TiffFile(self._filepath) as tif:
            images = tif.asarray()
            for page in tif.pages():
                for tag in page.tags.values():
                    _ = tag.name, tag.value
                image = page.asarray()

    def merge_duplicate_particles(self):
        """
        Merge particles with matching ID's
        """
        unique_ids = self.unique_ids(counts=True)
        duplicate_ids = unique_ids[unique_ids['count'] > 1].index.tolist()

        for dup_id in duplicate_ids:
            dup_p = self.get_particle(dup_id)
            assert len(dup_p) > 1
            merged_contour, merged_bbox = merge_particles(dup_p)

            # Remove original particles
            for i in range(len(dup_p)):
                self._particles.remove(dup_p[i])

            # Add the merged particle
            self._add_particle(dup_id, merged_contour, merged_bbox, self.masked)

        # calculate particle density (# of particles / # of pixels in image) for filtered image
        num_particles = len(self.particles)
        particle_density = num_particles / self.raw.size
        percent_particles_idd = num_particles / self.true_num_particles

        self._update_processing_stats(['particle_density', 'percent_particles_idd'],
            [particle_density, percent_particles_idd])

    def merge_overlapping_particles(self, cnts, bboxes, overlap_thresh=0.3, timeout=10):
        grp = 0
        grp_info = {}
        t0 = time.time()
        for bbox_cnt in sorted(zip(cnts, bboxes), key=lambda b: b[1][2] * b[1][3], reverse=True):
            cnt = bbox_cnt[0]
            bbox = bbox_cnt[1]
            if len(grp_info) == 0:
                grp_info.update({grp: (bbox, cnt)})
                grp += 1
            else:
                # Compute relative overlap with other bboxes and assign to group if overlap exceeds threshold
                new_contour = None
                new_bbox = None
                for grp_nr, grp_bbox_cnt in grp_info.items():
                    grp_bbox, grp_cnt = grp_bbox_cnt
                    rel_overlap = _compute_rel_bbox_overlap(bbox, grp_bbox)
                    if rel_overlap > overlap_thresh:
                        new_contour = cv2.convexHull(np.vstack([grp_cnt, cnt]))
                        new_bbox = cv2.boundingRect(new_contour)
                        break

                # If an overlapping contour was found, merge and update bounding box
                if (new_contour is not None) and (new_bbox is not None):
                    grp_info.update({grp_nr: (new_bbox, new_contour)})
                # If not overlapping bounding box was found, create a new group from this bounding box and contour
                else:
                    grp_info.update({grp: (bbox, cnt)})
                    grp += 1
            if timeout is not None:
                t_elapsed = time.time() - t0
                if t_elapsed > timeout:
                    logger.warning("Merging of overlapping particles reached timeout after {} s. "
                                   "Returning whatever was merged so far.".format(timeout))
                    break

        merged_bboxes = [el[0] for el in grp_info.values()]
        merged_cnts = [el[1] for el in grp_info.values()]

        return merged_cnts, merged_bboxes

    def pad_and_center_region(self, cX, cY, bbox, padding=2):
        """
        Steps:
            1. Pad and size the bounding box to be a square of odd-numbered side lengths.
            2. Center the bounding box on the computed contour center.
        Parameters
        ----------
        contour
        bbox
        padding

        Returns
        -------

        """
        # make the bounding box a square (w = h)
        if bbox[2] > bbox[3]:
            bbox = (bbox[0], bbox[1], bbox[2], bbox[2])
        if bbox[3] > bbox[2]:
            bbox = (bbox[0], bbox[1], bbox[3], bbox[3])

        # make the bounding box dimensions odd (w, h == odd number) to center the particle image on center pixel
        assert bbox[2] == bbox[3]
        if bbox[2] % 2 == 0:
            bbox = (bbox[0], bbox[1], bbox[2] + 1, bbox[3] + 1)

        # pad the bounding box to ensure the entire particle is captured
        bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding * 2, bbox[3] + padding * 2]

        # center bounding box on computed contour center
        bbox = [cX - int(np.floor(bbox[2]/2)), cY - int(np.floor(bbox[3]/2)), bbox[2], bbox[3]]

        return bbox, (cX, cY)

    def pad_and_center_contour(self, contour, bbox, padding=2):
        """
        Steps:
            1. Pad and size the bounding box to be a square of odd-numbered side lengths.
            2. Center the bounding box on the computed contour center.
        Parameters
        ----------
        contour
        bbox
        padding

        Returns
        -------

        """
        # make the bounding box a square (w = h)
        if bbox[2] > bbox[3]:
            bbox = (bbox[0], bbox[1], bbox[2], bbox[2])
        if bbox[3] > bbox[2]:
            bbox = (bbox[0], bbox[1], bbox[3], bbox[3])

        # make the bounding box dimensions odd (w, h == odd number) to center the particle image on center pixel
        assert bbox[2] == bbox[3]
        if bbox[2] % 2 == 0:
            bbox = (bbox[0], bbox[1], bbox[2] + 1, bbox[3] + 1)

        # pad the bounding box to ensure the entire particle is captured
        bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding * 2, bbox[3] + padding * 2]

        # compute center from contour
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"]) # note, x is in plotting coordinates (x would be columns in array)
        cY = int(M["m01"] / M["m00"]) # note, y is in plotting coordinates (y would be rows in array)

        # center bounding box on computed contour center
        bbox = [cX - int(np.floor(bbox[2]/2)), cY - int(np.floor(bbox[3]/2)), bbox[2], bbox[3]]

        return bbox, (cX, cY)

    def remove_nearby_particles(self, cnts, bboxes, threshold_distance=10):

        for bbox_cnt in sorted(zip(cnts, bboxes), key=lambda b: b[1][2] * b[1][3], reverse=True):
            cnt = bbox_cnt[0]
            bbox = bbox_cnt[1]
            cx = float(bbox[2] - bbox[0])
            cy = float(bbox[3] - bbox[1])

    def infer_self_similarity(self, function='sknccorr'):

        sim_func = sk_norm_cross_correlation

        sims = []
        for p_image in self.particles:
            img = p_image.template

            for p_template in self.particles:

                if p_template.id == p_image.id:
                    continue

                padx, pady = 0, 0
                temp = p_template.template
                imgx, imgy = img.shape
                tempx, tempy = temp.shape

                if tempx > imgx:
                    padx = tempx - imgx + 1
                if tempy > imgy:
                    pady = tempy - imgy + 1

                padded_img = np.pad(img, pad_width=[padx, pady], mode='constant', constant_values=np.min(img))

                sim = sim_func(padded_img, temp)
                sims.append(sim)

        mean_sim = np.mean(sims)

        self._particle_similarity = mean_sim

        return mean_sim

    def particle_coordinates(self, id_=None, skip_id_=None):
        coords = []
        no_z_count = 0
        for particle in self.particles:
            if id_ is not None:
                assert isinstance(id_, list)
                if particle.id not in id_:
                    continue
                if particle.id in skip_id_:
                    continue
            x, y = particle.location
            if particle.z is None:
                no_z_count += 1
                coords.append(pd.DataFrame({'id': [int(particle.id)], 'x': [x], 'y': [y], 'z': [np.nan]}))
            else:
                z = particle.z
                coords.append(pd.DataFrame({'id': [int(particle.id)], 'x': [x], 'y': [y], 'z': [z]}))
        if no_z_count > 0:
            if id_ is not None:
                total_particles = len(id_)
            else:
                total_particles = len(self.particles)
            logger.warning("Image {}: {} out of {} particles have no z coordinate".format(self.filename, no_z_count,
                                                                                          total_particles))
        if len(coords) != 0:
            coords = pd.concat(coords).sort_values(by='id')
        else:
            coords = pd.DataFrame()

        return coords

    def maximum_cm(self, id_=None):
        cms = []
        for particle in self.particles:
            if id_ is not None:
                assert isinstance(id_, list)
                if particle.id not in id_:
                    continue
            cm = particle.max_sim
            cms.append(pd.DataFrame({'id': [int(particle.id)], 'Cm': [cm]}))
        cms = pd.concat(cms).sort_values(by='id')

        return cms

    def set_true_z(self, z):
        assert isinstance(z, float)

        # set the z-value of the image
        self._z = z

        # If the image is set to be at a certain height, all the particles' true_z are assigned that height
        for particle in self.particles:
            particle.set_true_z(z=z)


    def set_z(self, z):
        """
        This sets both the particles' true_z and z value to input: z.
        """
        assert isinstance(z, float)

        # set the z-value of the image
        self._z = z

        # If the image is set to be at a certain height, all the particles' true_z are assigned that height
        for particle in self.particles:

            # only set particle true_z-coordinate if it hasn't already been set
            if particle.z_true is None:
                particle.set_true_z(z=z)

            # only set particle z-coordinate if it hasn't already been set
            if particle.z is None:
                particle.set_z(z)

    def add_particles_in_image(self):
        for p in self.particles:
            p.add_particle_in_image(img_id=self.z)

    def set_true_num_particles(self, num=None, data=None):
        if data is not None:
            self._true_num_particles = len(data)
        elif num is not None:
            self._true_num_particles = num
        else:
            raise ValueError("num and data cannot both be None. num or data must contain true number of particles")

    def unique_ids(self, counts=True):
        unique_ids = pd.DataFrame()
        if len(self.particles) > 0:
            for particle in self.particles:
                if particle.id not in unique_ids.index:
                    unique_ids = pd.concat([unique_ids, pd.DataFrame({'count': [1]}, index=[particle.id])])
                else:
                    unique_ids.loc[particle.id] = unique_ids.loc[particle.id] + 1

            unique_ids.index.name = 'particle_id'
        else:
            unique_ids = pd.DataFrame({}, columns=['count'])
            unique_ids.index.name = 'particle_id'
        if not counts:
            return unique_ids.index.tolist()
        else:
            return unique_ids

    @property
    def filename(self):
        return self._filename

    @property
    def filepath(self):
        return self._filepath

    @property
    def subbg(self):
        return self._subbg

    @property
    def filtered(self):
        return self._filtered

    @property
    def masked(self):
        return self._masked

    @property
    def original(self):
        return self._original

    @property
    def particles(self):
        return self._particles

    @property
    def processed(self):
        return self._filtered

    @property
    def raw(self):
        return self._raw

    @property
    def shape(self):
        return self.raw.shape

    @property
    def stats(self):
        return self._processing_stats

    @property
    def true_num_particles(self):
        return self._true_num_particles

    @property
    def z(self):
        return self._z

    @property
    def particle_similarity(self):
        return self._particle_similarity


def _compute_rel_bbox_overlap(bbox1, bbox2):
    # Convert bounding box (x, y, w, h) into rectangle (x0, y0, x1, y1)
    a_rect1 = bbox1[2] * bbox1[3]
    a_rect2 = bbox2[2] * bbox2[3]
    rect1 = bbox1[:2] + (bbox1[0] + bbox1[2], bbox1[1] + bbox1[3])
    rect2 = bbox2[:2] + (bbox2[0] + bbox2[2], bbox2[1] + bbox2[3])

    dx = min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])
    dy = min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])
    if (dx >= 0) and (dy >= 0):

        return dx * dy / min(a_rect1, a_rect2)
    else:
        return 0

def apply_filter(img, func, *args, **kwargs):
    assert callable(func)
    return func(img, *args, **kwargs)