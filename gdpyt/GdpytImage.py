import cv2
import numpy as np
import pandas as pd
from .preprocessing import apply_filter
from .particle_identification import apply_threshold, identify_contours, identify_circles, merge_particles
from .GdpytParticle import GdpytParticle
from os.path import isfile
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
        self._filename = path.split('\\')[-1]

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
        self._particles.append(GdpytParticle(self._filtered, id_, contour, bbox))

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

    def draw_particles(self, raw=True, contour_color=(0, 255, 0), thickness=2, draw_id=True, draw_bbox=True):
        if raw:
            canvas = self._raw.copy()
        else:
            canvas = self._filtered.copy()

        for particle in self.particles:
            cv2.drawContours(canvas, [particle.contour], -1, contour_color, thickness)
            if draw_id:
                bbox = particle.bbox
                coords = (int(bbox[0] - 0.2 * bbox[2]), int(bbox[1] - 0.2 * bbox[3]))
                cv2.putText(canvas, "ID: {}".format(particle.id), coords, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
            if draw_bbox:
                x, y, w, h = particle.bbox
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return canvas

    def filter_image(self, filterspecs):
        """
        This is basically you image_smoothing function. The argument filterdict are similar to the arguments of the
        image_smoothing function.
        e.g. filterdict: {'median': 5, 'bilateral': 4, 'gaussian': 5} or something along those lines
        :param filterdict:
        :return:

        This method should assign self._processed and self._processing_stats
        """
        img = (self._raw / 256).astype(np.uint8)
        for process_func in filterspecs.keys():
            func = eval(process_func)
            args = filterspecs[process_func]['args']
            if 'kwargs' in filterspecs[process_func].keys():
                kwargs = filterspecs[process_func]['kwargs']
            else:
                kwargs = {}

            img = apply_filter(img, func, *args, **kwargs)

        self._filtered = img.astype(np.uint8)
        self._histogram_preprocessed = cv2.calcHist([img], [0], None, [256], [0, 256])

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

    def identify_particles(self, thresh_specs, min_size=None, max_size=None, find_circles=False):

        if not find_circles:
            particle_mask = apply_threshold(self.filtered, parameter=thresh_specs)
            # Identify particles
            contours, bboxes = identify_contours(particle_mask)
            particle_mask = particle_mask.astype(bool)
            # masked_img = cv2.bitwise_and(self.filtered, self.filtered, mask=particle_mask)

            # Inverse of the particle area
            inv_mask = particle_mask != 0
            masked_img_inv = 255 * np.ones_like(particle_mask, dtype=np.uint8)
            masked_img_inv[inv_mask] = 0
            particle_mask_inv = cv2.bitwise_and(self.filtered, self.filtered, mask=masked_img_inv).astype(bool)

            # Calculate SNR + Particle image density
            snr_filt = self.filtered[particle_mask].mean() / self.filtered[particle_mask_inv].std()
            snr_raw = self.raw[particle_mask].mean() / self.raw[particle_mask_inv].std()
            p_density = particle_mask.sum() / particle_mask.size

            self._update_processing_stats(['snr_r', 'snr_f', 'rho_p'], [snr_raw, snr_filt, p_density])

        else:
            #_, particle_mask = apply_threshold(self.filtered)
            #masked_img = cv2.bitwise_and(self.filtered, self.filtered, mask=particle_mask)
            contours, bboxes = identify_circles(self.filtered)
        id_ = 0
        # Sort contours and bboxes by x-coordinate:
        for cont_bbox in sorted(zip(contours, bboxes), key=lambda b: b[1][0], reverse=True):
            contour = cont_bbox[0]
            contour_area = abs(cv2.contourArea(contour))
            # If specified, check if contour is too small or too large. If true, skip the creation of the particle
            if min_size is not None:
                if contour_area < min_size:
                    continue
            if max_size is not None:
                if contour_area > max_size:
                    continue

            bbox = cont_bbox[1]
            self._add_particle(id_, contour, bbox)
            id_ += 1

    def load(self, path, brightness_factor=100):
        img = cv2.imread(self._filepath, cv2.IMREAD_UNCHANGED)
        self._raw = img * brightness_factor
        # Calculate histogram
        self._histogram = cv2.calcHist([self._raw], [0], None, [256], [0, 256])

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
                coords.append(pd.DataFrame({'id': [particle.id], 'x': [x], 'y': [y]}))
            else:
                z = particle.z
                coords.append(pd.DataFrame({'id': [particle.id], 'x': [x], 'y': [y], 'z': [z]}))
        if no_z_count > 0:
            if id_ is not None:
                total_particles = len(id_)
            else:
                total_particles = len(self.particles)
            logger.warning("Image {}: {} out of {} particles have no z coordinate".format(self.filename, no_z_count,
                                                                                          total_particles))
        coords = pd.concat(coords).sort_values(by='id')

        return coords

    def set_z(self, z):
        assert isinstance(z, float)
        self._z = z
        # If the image is set to be at a certain height, all the particles in it are assigned that height
        for particle in self.particles:
            particle.set_z(z)

    def unique_ids(self, counts=True):
        unique_ids = pd.DataFrame()
        for particle in self.particles:
            if particle.id not in unique_ids.index:
                unique_ids = pd.concat([unique_ids, pd.DataFrame({'count': [1]}, index=[particle.id])])
            else:
                unique_ids.loc[particle.id] = unique_ids.loc[particle.id] + 1

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
