import cv2
import numpy as np
import pandas as pd

class GdpytParticle(object):

    def __init__(self, image, id_, contour, bbox):
        super(GdpytParticle, self).__init__()
        self._id = id_
        assert isinstance(image, np.ndarray)
        self._image = image
        self._contour = contour
        self._set_bbox(bbox)
        self._compute_center()
        self._compute_convex_hull()
        self._similarity_curve = None
        self._z = None

    def _create_template(self):
        x, y, w, h = self._bbox
        self._template = self._image[y: y + h, x: x + w]

    def _compute_convex_hull(self):
        hull = cv2.convexHull(self.contour)
        self._hull = hull
        self._area = float(cv2.contourArea(hull))
        self._solidity = cv2.contourArea(self.contour) / self._area

    def _compute_center(self):
        M = cv2.moments(self._contour)

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        self._set_location((cX, cY))

    def _set_location(self, location):
        assert len(location) == 2
        self._location = location

    def _set_bbox(self, bbox):
        self._bbox = bbox
        self._create_template()

    def resize_bbox(self, w, h):
        """
        Adjust bounding box to size w x h and adjust the center to the center of the contour
        :param w: new width (int)
        :param h: new height (int)
        :return:
        """
        wl, ht = int(w / 2), int(h / 2)
        top_corner = np.array(self.location) - np.array([wl, ht])
        self._set_bbox((top_corner[0], top_corner[1], w, h))

    def set_similarity_curve(self, z, sim, label_suffix=None):
        assert len(z) == len(sim)
        columns = ['z', 'S_{}'.format(label_suffix.upper())]
        self._similarity_curve = pd.DataFrame({columns[0]: z, columns[1]: sim})

    def set_z(self, z):
        assert isinstance(z, float)
        self._z = z

    def set_id(self, id_):
        self._id = id_

    @property
    def area(self):
        return self._hull

    @property
    def bbox(self):
        return self._bbox

    @property
    def contour(self):
        return self._contour

    @property
    def hull(self):
        return self._hull

    @property
    def id(self):
        return self._id

    @property
    def location(self):
        return self._location

    @property
    def solidity(self):
        return self._solidity

    @property
    def similarity_curve(self):
        return self._similarity_curve

    @property
    def template(self):
        return self._template

    @property
    def z(self):
        return self._z