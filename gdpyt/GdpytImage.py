import cv2
from skimage.filters import median, gaussian
from skimage.morphology import disk, white_tophat
import numpy as np
import pandas as pd
from .particle_identification import apply_threshold, identify_contours, identify_circles, merge_particles
from .GdpytParticle import GdpytParticle
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

    def __init__(self, path):
        super(GdpytImage, self).__init__()
        # Attributes with an underscore as the first character are "internal use". That means that they are only meant
        # to be modified by methods of this class.
        # The reasoning behind this is the following: Imagine an image with a number of particles. The particles are saved
        # in the self._particles attribute. We only want the identify_particles method in this class to modify this attribute.
        # We don't really want any other part of the program to change this attribute, since the particles in an image
        # are always going to be those that were identified with identify_particles. If this attribute isn't marked as
        # "internal use", other parts of the program could by accident add or delete particles.

        if not isfile(path):
            raise ValueError("{} is not a valid file".format(path))

        self._filepath = path
        self._filename = basename(path)

        # Load the image. This sets the ._raw attribute
        self.load(path)

        # Filtered image. This attribute is assigned by using the filter_image method
        self._filtered = None
        self._processing_stats = None
        self._masked = None

        # Particles: dictionary {particle_id: Particle object}
        # This dictionary is filled up with the identify_particles method
        self._particles = []
        self._z = None

    def __repr__(self):
        class_ = 'GdpytImage'
        repr_dict = {'Dimensions': self.shape,
                     'Particles in this image': [particle.id for particle in self.particles],
                     'Z coordinate': self._z}
        out_str = "{}: {} \n".format(class_, self.filename)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def _add_particle(self, id_, contour, bbox):
        self._particles.append(GdpytParticle(self._raw, self._filtered, id_, contour, bbox))

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
            canvas = self._raw.copy()
        else:
            canvas = self._filtered.copy()

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

    def filter_image(self, filterspecs, force_rawdtype=True):
        """
        This is an image filtering function. The argument filterdict are similar to the arguments of the
        image_smoothing function.
        e.g. filterdict: {'median': 5, 'bilateral': 4, 'gaussian': 5}
        :param filterdict:
        :return:

        This method should assign self._processed and self._processing_stats
        """
        valid_filters = ['median', 'gaussian', 'white_tophat', 'equalize_adapthist']

        # Convert to 8 byte uint for filter operations
        img_copy = self._raw.copy()
        raw_dtype = img_copy.dtype

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

    def identify_particles(self, thresh_specs, min_size=None, max_size=None, shape_tol=0.1):
        if shape_tol is not None:
            assert 0 < shape_tol < 1
        particle_mask = apply_threshold(self.filtered, parameter=thresh_specs).astype(np.uint8)
        self.masked = particle_mask
        # Identify particles
        contours, bboxes = identify_contours(particle_mask)
        logger.debug("{} contours in thresholded image".format(len(contours)))
        contours, bboxes = self.merge_overlapping_particles(contours, bboxes)
        logger.debug("{} contours in thresholded image after merging of overlapping".format(len(contours)))

        id_ = 0
        # Sort contours and bboxes by x-coordinate:
        skipped_cnts = []
        for cont_bbox in sorted(zip(contours, bboxes), key=lambda b: b[1][0], reverse=True):
            contour = cont_bbox[0]
            contour_area = abs(cv2.contourArea(contour))
            # get perimeter
            contour_perim = cv2.arcLength(contour, True)

            # If specified, check if contour is too small or too large. If true, skip the creation of the particle
            if min_size is not None:
                if contour_area < min_size:
                    skipped_cnts.append(contour)
                    continue
            if max_size is not None:
                if contour_area > max_size:
                    skipped_cnts.append(contour)
                    continue

            bbox = cont_bbox[1]

            if shape_tol is not None:
                # Discard contours that are clearly not a circle just by looking at the aspect ratio of the bounding box
                bbox_ar = bbox[2] / bbox[3]
                if abs(np.maximum(bbox_ar, 1 / bbox_ar) - 1) > shape_tol:
                    skipped_cnts.append(contour)
                    continue
                # Check if circle by calculating thinness ratio
                tr = 4 * np.pi * contour_area / contour_perim**2
                logging.debug("bbox_ar: {}, circ {}".format(bbox_ar, tr))
                if abs(np.maximum(tr, 1 / tr) - 1) > shape_tol:
                    skipped_cnts.append(contour)
                    continue

            # Add particle
            self._add_particle(id_, contour, bbox)
            id_ += 1

        # Fill in areas of the skipped particles in the particle mask
        #for i in range(len(skipped_cnts)):
            #cv2.drawContours(particle_mask, skipped_cnts, i, color=0, thickness=-1)

        particle_mask = particle_mask.astype(bool)
        # masked_img = cv2.bitwise_and(self.filtered, self.filtered, mask=particle_mask)

        # Inverse of the particle area
        inv_mask = particle_mask != 0

        # Calculate SNR + Particle image density
        sigma_bckgr = self.raw[inv_mask].std()
        sigma_bckgr_f = self.filtered[inv_mask].std()
        mean_bckgr_r = self.raw[inv_mask].mean()
        mean_bckgr_filt = self.filtered[inv_mask].mean()
        snr_filt = (self.filtered[particle_mask].mean() - mean_bckgr_filt)/ self.filtered[inv_mask].std()
        snr_raw = (self.raw[particle_mask].mean() - mean_bckgr_r) / self.raw[inv_mask].std()
        p_density = particle_mask.sum() / particle_mask.size

        self._update_processing_stats(['mean_bckgr_r', 'mean_bckgr_f', 'sigma_bckgr_r', 'sigma_bckgr_f', 'snr_r', 'snr_f', 'rho_p'],
                                      [mean_bckgr_r, mean_bckgr_filt, sigma_bckgr, sigma_bckgr_f, snr_raw, snr_filt, p_density])

    def is_infered(self):
        return all([particle.z is not None for particle in self.particles])

    def load(self, path, mode=cv2.IMREAD_UNCHANGED):
        img = cv2.imread(self._filepath, mode)
        self._raw = img # cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def merge_duplicate_particles(self):
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
            self._add_particle(dup_id, merged_contour, merged_bbox)

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


    def particle_coordinates(self, id_=None):
        coords = []
        no_z_count = 0
        for particle in self.particles:
            if id_ is not None:
                assert isinstance(id_, list)
                if particle.id not in id_:
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

    def set_z(self, z):
        assert isinstance(z, float)
        self._z = z
        # If the image is set to be at a certain height, all the particles in it are assigned that height
        for particle in self.particles:
            particle.set_z(z)

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
    def filtered(self):
        return self._filtered

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