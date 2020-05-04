import cv2
import numpy as np
from .preprocessing import apply_filter
from .particle_identification import apply_threshold, identify_contours, identify_circles
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

    @property
    def raw(self):
        return self._raw

    @property
    def processed(self):
        return self._filtered

    @property
    def filename(self):
        return self._filename

    @property
    def filepath(self):
        return self._filepath

    def load(self, path, brightness_factor=100):
        img = cv2.imread(self._filepath, cv2.IMREAD_UNCHANGED)
        self._raw = img * brightness_factor
        # Calculate histogram
        self._histogram = cv2.calcHist([self._raw], [0], None, [256], [0, 256])

    def set_z(self, z):
        assert isinstance(z, float)
        self._z = z
        # If the image is set to be at a certain height, all the particles in it are assigned that height
        for particle in self.particles:
            particle.set_z(z)

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

    def identify_particles(self, min_size=None, max_size=None, find_circles=False):
        particles = []

        if not find_circles:
            _, particle_mask = apply_threshold(self.filtered)
            masked_img = cv2.bitwise_and(self.filtered, self.filtered, mask=particle_mask)
            contours, bboxes = identify_contours(particle_mask)
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
            particles.append(GdpytParticle(self._raw, id_, contour, bbox))
            id_ += 1

        self._particles = particles

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
                cv2.putText(canvas, "ID: {} ({}, {})".format(particle.id, particle.location[0], particle.location[1]), coords, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
            if draw_bbox:
                x, y, w, h = particle.bbox
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return canvas

    def get_particle(self, id):
        for particle in self.particles:
            if particle.id == id:
                return particle
        logger.error("No particle with ID {} found in this image".format(id))

    def shape(self):
        return self.raw.shape

    @property
    def particles(self):
        return self._particles

    @property
    def filtered(self):
        return self._filtered
